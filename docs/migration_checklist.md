# Migration checklist

1. Server metadata 的 `action_schema` 升级为 `sharpa_policy_action.v4`。
2. 每个 model adapter 填满 output 的全部最小 key，将 relative EEF 转为 absolute，
   无预测的 `auxiliary` value 填 `None`。
3. Workstation ROS 只保留一个 `policy_client`，接入七路 producer 和
   `PolicyClientCore`。
4. 将原执行调度 node 的 server 请求逻辑删除；执行职责迁入 `action_execute`。
5. `action_ik` 和 `action_execute` 全程透传 request/action/revision/execution identity。
6. 在 PND 和 UR 分别实现本体坐标、IK、ACK、TTL 和 safe-stop。
7. 先 shadow 验证 observation/action/done，再启用真机 command。
8. 两个本体均运行 `PYTHONPATH=interface python3 -m pytest contract_tests -q`。

当前第 1、3、4 步已经完成：BAAI template、PND workstation 和 UR client 均使用
v4 action；旧 v3 action client 不能与新 server 混用。
