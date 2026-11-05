"""Structured, latched operator diagnostics for execution protection."""
import time


class ProtectionError(RuntimeError):
    def __init__(self, code, message, **context):
        super().__init__(message)
        self.code, self.context = code, context


TITLES = {
    'hand_range_rejected': '手部绝对关节越界',
    'hand_projection_too_large': '手部目标范围修正过大',
    'startup_pose_incompatible': '手部启动姿态偏差过大',
    'hand_command_jump': '手部命令变化过大',
    'hand_tracking_hard_stop': '手部跟踪误差过大',
    'hand_tracking_timeout': '手部跟踪误差持续超限',
    'arm_command_jump': '机械臂命令变化过大',
    'arm_startup_pose_incompatible': '机械臂启动姿态偏差过大',
    'arm_tracking_error': '机械臂跟踪误差过大',
    'arm_absolute_range_rejected': '机械臂绝对关节越界',
    'chunk_command_jump': '策略动作序列变化过大',
    'firmware_limits_mismatch': '手部设备与固件限位表不匹配',
    'observation_timestamp_skew': '相机与关节状态不同步',
    'observation_timestamp_stale': '观测过期或时间戳异常',
    'observation_invalid': '相机或机器人状态无效',
    'hil_input_stale': 'HIL 人工输入缺失或过期',
    'eef_ik_rejected': '末端位姿逆解被拒绝',
    'recorder_unavailable': '数据采集未就绪或中断',
    'model_unavailable': '模型退出或状态心跳超时',
    'inference_timeout': '推理通信超时',
    'invalid_action': '动作维度或数值无效',
    'hardware_rejected': '硬件执行拒绝',
    'ur_hardware_protection': 'UR 硬件状态异常或保护停机',
    'arbitration_rejected': '控制权或执行状态不允许',
    'execution_error': '执行异常',
}
GROUPS = {'left_hand': '左手', 'right_hand': '右手', 'left_arm': '左臂', 'right_arm': '右臂'}


def protection_event(reason, *, source='actor', hardware_code=None, hil=None):
    message = str(reason)
    code = getattr(reason, 'code', '')
    if not code or code == 'execution_error':
        code = next((key for key in TITLES if key != 'execution_error' and key in message), '')
    if not code:
        patterns = (
            ('rollout record', 'recorder_unavailable'),
            ('model unloaded', 'model_unavailable'), ('normal running mode', 'ur_hardware_protection'),
            ('joint target exceeds absolute', 'arm_absolute_range_rejected'),
            ('joint target step exceeds', 'arm_command_jump'),
            ('joint limiter', 'arm_command_jump'), ('finite', 'invalid_action'),
            ('invalid action chunk', 'invalid_action'),
        )
        code = next((key for text, key in patterns if text in message), '')
    if not code and isinstance(reason, TimeoutError):
        code = 'inference_timeout'
    if not code and type(reason).__name__ == 'ObservationUnavailable':
        code = 'observation_invalid'
    code = code or (hardware_code if hardware_code in TITLES else 'hardware_rejected' if hardware_code else 'execution_error')
    rows = getattr(reason, 'violations', [])
    details = []
    for row in rows:
        label = f"{GROUPS.get(row['group'], row['group'])}关节 {row['joint_number']}"
        if 'lower_rad' in row:
            details.append(f"{label}：目标 {row['target_rad']:.6f} rad；允许范围 [{row['lower_rad']:.6f}, {row['upper_rad']:.6f}] rad")
        else:
            details.append(f"{label}：参考/实测 {row['measured_rad']:.6f} → 目标 {row['target_rad']:.6f} rad；差值 {row['delta_rad']:+.6f}，阈值 {row['limit_rad']:.6f} rad")
    diagnostics = getattr(reason, 'diagnostics', {})
    if diagnostics:
        if code == 'observation_timestamp_skew':
            details.append(f"时间差 {diagnostics['skew_ms']:.1f} ms；上限 {diagnostics['max_skew_ms']:.1f} ms；最旧输入 {diagnostics['oldest_source']}")
        for name, row in diagnostics.get('sources', {}).items():
            age = row.get('age_ms')
            details.append(f"{name} 数据年龄：{age:.1f} ms；上限 {diagnostics['max_age_ms']:.1f} ms" if age is not None else f'{name} 时间戳无效')
    context = getattr(reason, 'context', {})
    if code == 'eef_ik_rejected' and context.get('ik_rejection'):
        rejection = context['ik_rejection']
        stage = rejection.get('stage', '')
        label = {'WRIST_FLIP_HOLD': '换腕/关节跳变保护',
                 'JOINT_POSITION_LIMIT': '关节绝对限位',
                 'IKFAST_NO_SOLUTION': '未找到可用逆解'}.get(stage, stage)
        side = GROUPS.get(context.get('side', '') + '_arm', context.get('side', ''))
        details.append(f"{side}，预测动作第 {context['chunk_step']+1} 步：{label}")
        if stage == 'WRIST_FLIP_HOLD':
            details.append(f"候选解最小最大关节变化 {rejection['minimum_max_delta_rad']:.6f} rad；"
                           f"阈值 {rejection['flip_threshold_rad']:.6f} rad")
    if code == 'hil_input_stale':
        limit = context.get('limit_s', .25)
        ages = context.get('input_ages', (hil or {}).get('input_ages', {}))
        for key in ('left_arm', 'right_arm', 'left_hand', 'right_hand'):
            age = ages.get(key)
            if age is None or age > limit:
                details.append(f"{GROUPS[key]}：未收到输入；上限 {limit*1000:.0f} ms" if age is None else
                               f"{GROUPS[key]}：输入年龄 {age*1000:.1f} ms；上限 {limit*1000:.0f} ms")
    # Preserve the original message for driver errors whose ROS ACK has no structured rows.
    return dict(active=True, code=code, title=TITLES.get(code, '执行异常'),
                source=source, timestamp_ns=time.time_ns(), message=message, details=details,
                violations=rows, observation_diagnostics=diagnostics, context=context,
                hardware_code=hardware_code, hil=hil or {})
