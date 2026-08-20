# SharpA dict interface

Interface dict 是 server、workstation 和 embodiment adapter 的唯一边界。所有下列
key 都必须存在，允许添加额外 key；只有标注 `| None` 的 value 可以为空。

## MetadataFormat

```python
{
    "schema": "sharpa_policy_metadata_format.v1",
    "format_id": str,
    "image": {
        "ego_cam": {"history_len": int, "current": bool},
        "left_wrist_cam": {"history_len": int, "current": bool},
        "right_wrist_cam": {"history_len": int, "current": bool},
    },
    "state": {
        "history_len": int,
        "current": bool,
        "left_wrist": {"joint": bool, "eef": bool},
        "right_wrist": {"joint": bool, "eef": bool},
        "hand_joint": {"left": bool, "right": bool},
    },
    "sensor": {
        "tau": {"history_len": int, "current": bool},
        "wrench": {"history_len": int, "current": bool},
        "deformation": {"history_len": int, "current": bool},
    },
}
```

`next_metadata_format=None` 表示沿用当前格式；非 None 时必须是完整格式。

## Policy input

```python
{
    "schema": "sharpa_policy_observation.v3",
    "metadata_format_id": str,
    "session_id": str,
    "request_id": int,
    "timestamp_ns": int,
    "prompt": str,
    "image": {
        "ego_cam": {"history": list, "current": CameraFrame | None},
        "left_wrist_cam": {"history": list, "current": CameraFrame | None},
        "right_wrist_cam": {"history": list, "current": CameraFrame | None},
    },
    "state": {
        "history": StateBatch | None,
        "current": StateFrame | None,
    },
    "sensor": {
        "tau": {"history": SensorBatch | None, "current": SensorFrame | None},
        "wrench": {"history": SensorBatch | None, "current": SensorFrame | None},
        "deformation": {"history": SensorBatch | None, "current": SensorFrame | None},
    },
    "execution_feedback": {
        "last_action_id": str | None,
        "executed_steps": int,
        "success": bool,
    },
}
```

State wrist 始终包含 `joint`、`eef`、`eef_def`。`eef_def=None` 等价于默认
`absolute`；public interface 不接收 relative observation。EEF 使用
`xyz + rot6d`，shape `(9,)`，具体本体坐标由 ROS adapter 负责。

固定数据：hand joint/tau 每手 22D；wrench 每手 `(5,6)`；deformation 每手
`uint8 (5,240,240)`；相机使用 JPEG bytes。History 不含 current。

## Policy output

```python
{
    "schema": "sharpa_policy_action.v4",
    "session_id": str,
    "request_id": int,
    "action_id": str,
    "revision": int,
    "timestamp_ns": int,
    "execution": {
        "frequency_hz": float,
        "action_length": int,
        "execute_start": int,
        "execute_length": int,
    },
    "action": {
        "left_wrist": {
            "joint": np.ndarray | None,
            "eef": np.ndarray | None,
            "eef_def": "absolute" | None,
        },
        "right_wrist": {
            "joint": np.ndarray | None,
            "eef": np.ndarray | None,
            "eef_def": "absolute" | None,
        },
        "hand_joint": {
            "left": np.ndarray | None,
            "right": np.ndarray | None,
        },
    },
    "auxiliary": {
        "video": {
            "ego": None,
            "left_wrist": None,
            "right_wrist": None,
        },
        "tactile": {
            "deformation": None,
            "wrench": None,
            "hand_tau": None,
        },
    },
    "diagnostics": dict | None,
    "next_metadata_format": MetadataFormat | None,
}
```

Server policy 内部可以使用 relative EEF，但发送到 workstation 前必须转换为
`absolute`；`parse_policy_action()` 默认拒绝 public boundary 上的 relative EEF。
当前 auxiliary 所有值填 `None`，以后支持预测模态时保留同名 key。

## ExecutableAction

Policy client 按 execution slice 截断数组后发给 `action_ik`。跨 node 时必须保留
identity，`action_ik` 只替换或补充 joint plan，不能修改这些字段：

```python
{
    "schema": "sharpa_executable_action.v1",
    "session_id": str,
    "request_id": int,
    "action_id": str,
    "revision": int,
    "timestamp_ns": int,
    "execution": {
        "frequency_hz": float,
        "action_length": int,   # 原始 chunk 长度
        "execute_start": int,
        "execute_length": int,  # 当前 action arrays 的第一维
    },
    "action": ActionSlice,
}
```

## ExecutionDone

`action_execute` 完成或终止一个 slice 时只发布一次：

```python
{
    "schema": "sharpa_execution_done.v1",
    "request_id": int,
    "action_id": str,
    "revision": int,
    "execute_start": int,
    "execute_length": int,
    "executed_steps": int,
    "success": bool,
    "done": True,
    "error": str | None,
}
```

成功时 `executed_steps == execute_start + execute_length`。Policy client 只接受与
当前 action 完全匹配的 done，随后把它转成下一次 input 的 execution feedback。
