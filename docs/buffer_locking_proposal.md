# interface buffer 加锁方案（待 PND 侧签字）

`interface/sharpa_interface/server/buffers.py` 没有任何锁。ROS 多线程 executor 下
producer 和 consumer 并发访问会撕裂数据。三条竞争已在
`contract_tests/test_buffer_concurrency.py` 复现，**确定性失败，不是偶发**。

改动落在共享的 `interface/`，影响 PND 和 UR 两个本体，需要两侧 owner 同意。

## 三条竞争不在同一层

| # | 竞争 | 位置 | 复现结果 |
|---|---|---|---|
| 1 | `snapshot()` 内 `list()` 拷贝期间发生淘汰 | `buffers.py` `snapshot()` | 交错 push 下 history 少一帧 |
| 2 | `snapshot()` 与 `latest()` 是两次独立读取 | `build_policy_input()` | 200/200 次 build 撕裂，tail 与 current 差 2 帧 |
| 3 | 七路 buffer 顺序读取，无一致性边界 | `build_policy_input()` | 单份 input 混入 2 个不同采集瞬间 |

第 3 条最严重：相机、手状态、触觉来自不同时刻，是物理上不存在的状态，且 shape 校验全过、
不报任何错。

**只给 `TemporalBuffer` 加锁解决不了 2 和 3** —— 它们跨多次调用，单 buffer 锁在每次调用
返回时就释放。

## 三层修复

### 第一层：`TemporalBuffer` 内部锁

`push` / `snapshot` / `current` / `latest` / `clear` / `__len__` 各自持锁。解决竞争 1
（`deque.append` 本身原子，但 `list(self._items)[-count:]` 不是）。

代价：每次读写一次锁获取。30Hz × 7 路可忽略。

### 第二层：一致读取 API

竞争 2 的根因是 history 和 current 分两次取。新增一个方法，在一次持锁读取内同时返回两者，
保证 current 不出现在 history 里——把「History 不含 current」从调用方自觉变成实现保证。

纯加法，不删旧 API，现有代码不改也不会坏。

### 第三层：跨 buffer 一致性

要求七路读取落在同一瞬间。三个选项：

- **A** `PolicyInputBuffers` 持一把共享锁，`build_policy_input()` 全程持有，producer 的
  `push` 也走这把锁
- **B** 乐观读 + 版本号校验，读完发现变了就重试
- **C** 双缓冲，producer 写影子副本，consumer 读稳定副本

## 结论：选 A，有实测数据

`build_policy_input()` 耗时（n=2000，真实 payload：JPEG 100KB × 3 路、deformation
`(2,5,240,240)` uint8、history_len=4）：

| 指标 | 值 | 占 30Hz 预算 |
|---|---|---|
| P50 | 0.0072 ms | 0.022% |
| P99 | 0.0193 ms | 0.058% |
| max | 0.1234 ms | 0.371% |

互斥窗口 0.02ms，producer 210 次/秒，撞上概率极低且撞上只等 0.02ms。A 不阻塞采集。

耗时这么低是因为**全程无数据拷贝**：JPEG 是 `bytes` 引用，deformation 是 ndarray 引用，
`snapshot()` 的 `list()` 只拷指针。耗时与 payload 大小无关——换更大的相机或更长 history
都不改变这个结论。

B 的重试在 210 次/秒写入下不确定收敛；C 的复杂度在 A 已经够快的前提下不值得。

## 验收

`contract_tests/test_buffer_concurrency.py` 三条从预期失败转为通过。三条分别对应三层修复，
可以分层上线、分层验证。

## 待确认

1. PND 侧是否真的在多线程 executor 下调 `build_policy_input()`？如果 PND 是单线程，
   这个改动对 PND 只是无害开销，签字应该容易。
2. 是否需要 schema version bump？我认为不需要——dict 形状和字段语义都不变，只是并发安全性
   修复。但这条归 interface owner 判断。
