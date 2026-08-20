# Contract tests

这些测试冻结 server/client 的最小可扩展 dict、metadata-driven buffers、action slice、
identity matching 和累计 execution feedback。测试不需要 ROS 或硬件。

```bash
PYTHONPATH=interface python3 -m pytest contract_tests -q
```

新增或修改 interface key、shape、执行语义时，必须同时增加一个能捕获协议漂移的测试。
