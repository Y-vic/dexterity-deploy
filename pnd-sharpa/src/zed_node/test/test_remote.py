from zed_node.remote import RemoteConfig, _pipeline_args


def test_default_pipeline_has_no_quest_stream() -> None:
    args = _pipeline_args(RemoteConfig(inference_stream_enabled=False))

    assert "rtph264pay" in args
    assert "udpsink" in args
    assert "tcpserversink" not in args
    assert "mpegtsmux" not in args


def test_recording_and_quest_streams_are_independent() -> None:
    args = _pipeline_args(
        RemoteConfig(
            monitor_stream_enabled=True,
            monitor_stream_host="10.10.20.127",
            monitor_stream_port=5600,
            inference_stream_enabled=False,
            quest_stream_enabled=True,
            quest_stream_bind_host="0.0.0.0",
            quest_stream_port=5602,
        )
    )

    assert args.count("rtph264pay") == 1
    assert args.count("udpsink") == 1
    assert "host=10.10.20.127" in args
    assert "port=5600" in args
    assert args.count("mpegtsmux") == 1
    assert args.count("tcpserversink") == 1
    assert "host=0.0.0.0" in args
    assert "port=5602" in args
    assert "sync-method=latest" in args
    assert "recover-policy=latest" in args
    assert "buffers-max=1024" in args
    assert "buffers-soft-max=256" in args
