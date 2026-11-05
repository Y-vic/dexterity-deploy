import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ae01_node import AE01Core, robot_joints
from ae01_node.ae01_node import output_topic
from ae01_node.sdk_worker import RAW_IMU_FIELDS, SdkReader, prepare_config, vendor_processes
from sharpa_interface.sharpa import JOINT_NAMES_PER_HAND


def sample(now=10.0):
    return dict(monotonic_s=now, sides={side: dict(online=True, ready=True,
        raw_seen_s=now, robot_seen_s=now, raw_timestamp_ns=100, robot_timestamp_ns=100,
        q=[.01*i for i in range(22)], names=JOINT_NAMES_PER_HAND)
        for side in ("left", "right")})


def test_robot_order_names_and_radian_values():
    values = [.01*i for i in range(22)]
    names = ["RIGHT_" + n.upper() for n in reversed(JOINT_NAMES_PER_HAND)]
    assert robot_joints(list(reversed(values)), names, "right") == values
    assert robot_joints(values) == values  # Documented fixed SDK ROBOT order.
    with pytest.raises(ValueError, match="names"):
        robot_joints(values, ["RAW_MCP_AA"]*22)
    for bad in ([0]*21, [0]*23, [float("nan")]*22, [float("inf")]*22):
        with pytest.raises(ValueError, match="finite"):
            robot_joints(bad)


def test_sdk_indexed_robot_names_map_to_documented_wave_order():
    values = [.01*i for i in range(22)]
    assert robot_joints(values, [f"joint_{i}" for i in range(22)]) == values
    names = [f"RIGHT_JOINT_{i}" for i in reversed(range(22))]
    assert robot_joints(list(reversed(values)), names, "right") == values
    with pytest.raises(ValueError, match="names"):
        robot_joints(values, [f"joint_{i}" for i in range(1, 23)])


def test_full_partial_disabled_and_worker_exit():
    core = AE01Core(clock=lambda: 10.0)
    assert core.command() is None
    packet = sample()
    core.observe(packet)
    q, left, right = core.command()
    assert left and right and q == packet["sides"]["left"]["q"]*2
    packet["sides"]["right"]["online"] = False
    core.observe(packet)
    q, left, right = core.command()
    assert left and not right and q[22:] == [0]*22
    assert core.command(alive=False) is None
    assert not core.status(alive=False)["valid"]


@pytest.mark.parametrize("field", ["raw_seen_s", "robot_seen_s"])
def test_either_stalled_source_expires_even_with_live_worker_packets(field):
    now = [10.0]
    core = AE01Core(clock=lambda: now[0])
    packet = sample()
    core.observe(packet)
    now[0] = 10.3
    packet["monotonic_s"] = now[0]
    for row in packet["sides"].values():
        row["raw_seen_s"] = row["robot_seen_s"] = now[0]
        row[field] = 10.0
    core.observe(packet)
    assert core.command() is None


@pytest.mark.parametrize("change", [
    lambda row: row.update(ready=False), lambda row: row.update(error="fault 0x0940"),
    lambda row: row.update(q=[float("nan")]*22),
    lambda row: row.update(raw_seen_s=float("nan")),
    lambda row: row.update(raw_seen_s=11.0),
])
def test_invalid_side_never_enables_other_side_or_fabricates_targets(change):
    core = AE01Core(clock=lambda: 10.0)
    packet = sample()
    change(packet["sides"]["left"])
    core.observe(packet)
    q, left, right = core.command()
    assert not left and right and q[:22] == [0]*22


def test_delayed_queue_packet_does_not_refresh_age_and_missing_side_is_cleared():
    core = AE01Core(clock=lambda: 11.0)
    core.observe(sample(10.0))
    assert core.command() is None
    packet = sample(11.0)
    packet["sides"].pop("right")
    core.observe(packet)
    assert core.command()[1:] == (True, False)
    with pytest.raises(ValueError, match="timestamp"):
        core.observe(sample(10.9))


def test_status_contains_source_timestamps_and_per_side_retarget_rates():
    core = AE01Core(clock=lambda: 10.1)
    packet = sample()
    packet["sides"]["left"].update(hz=150., robot_hz=60., sn="LEFT-SN")
    core.observe(packet)
    status = core.status(True, True)
    assert status["provider"] == "ae01" and status["valid"]
    assert status["left_robot_timestamp_ns"] == 100
    assert status["left_robot_age_s"] == pytest.approx(.1)
    assert status["left_robot_hz"] == 60 and status["left_sn"] == "LEFT-SN"


def test_authoritative_control_routes_teleop_and_hil_and_rejects_stale_gate():
    def state(source, hand=1):
        return (10., NS(active_source=source, sharpa_state=hand))
    assert output_topic(False, False, state(2), 10.1) == "/sharpa/command"
    assert output_topic(True, False, state(3), 10.1) == "/hil/hand_candidate"
    assert output_topic(True, True, state(2), 10.1) == "/sharpa/command"
    assert output_topic(True, True, state(3), 10.1) == "/hil/hand_candidate"
    for control in (None, state(1), state(2, 0), state(2, 2), state(2, 3), state(2, 4)):
        assert output_topic(False, False, control, 10.1) is None
    assert output_topic(False, False, state(2), 10.6) is None
    assert output_topic(True, False, state(2), 10.1) is None


@pytest.mark.parametrize("transport,expected", [("wireless", "usb_serial"), ("wired", "ethernet_udp"), ("usb", "usb_vendor")])
def test_config_explicit_transport_and_no_wave_without_editing_vendor_file(tmp_path, transport, expected):
    config = dict(enable_wave=True, enable_wave_rk=True, enable_localhost_glove=True,
                  transport_link_options=["unsafe"], casadi_common=dict(retarget_update_rate_hz=60))
    path = tmp_path/"sdk_config.json"
    original = json.dumps(config)
    path.write_text(original)
    resolved, result = prepare_config(path, transport)
    assert resolved == path and result["transport_link"] == expected
    assert all(result[key] is False for key in ("enable_wave", "enable_wave_rk", "wave_enabled", "enable_localhost_glove"))
    assert "transport_link_options" not in result
    assert path.read_text() == original
    config["casadi_common"]["retarget_update_rate_hz"] = 0
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="retarget"):
        prepare_config(path, transport)


class Device:
    def __init__(self):
        self.stamps = {"raw": 1, "robot": 1}
        self.events = []

    def init(self, config): self.events.append("init"); return 0
    def start(self): self.events.append("start"); return 0
    def stop(self): self.events.append("stop"); return 0
    def destroy(self): self.events.append("destroy")

    def fetch_data(self, kind):
        stamp = NS(sec=self.stamps[kind], nanosec=0)
        hand = NS(header=NS(stamp=stamp), joint=NS(position=[.1]*22, name=[]))
        return 0, NS(**{kind: hand})


def reader_fixture(clock, fail_init=False):
    devices = {"left": Device(), "right": Device()}
    infos = [NS(device_side=side, device_type="glove", online=True, sn=side, pn="AvatarExo-LL-02", hw_version="2.4.1A", hz=150.,
                robot_hz=60., retarget_bootstrap_ready=True, error_code=0, diagnostics={}) for side in devices]
    events = []
    sdk = NS(initialize=lambda *_: 1 if fail_init else 0,
             list_device_info=lambda: infos, get_device=lambda _, side: devices[side],
             destroy=lambda: events.append("sdk.destroy"))
    api = NS(AvatarSDK=NS(get_instance=lambda: sdk), ErrorCode=NS(SUCCESS=0),
             DeviceType=NS(GLOVE="glove"), DeviceSide=NS(LEFT="left", RIGHT="right"),
             DeviceDataCategery=NS(RAW="raw", ROBOT="robot"))
    return api, devices, infos, events


def test_sdk_cache_reads_do_not_count_as_new_samples_or_replay_on_reconnect():
    now = [10.0]
    api, devices, infos, events = reader_fixture(lambda: now[0])
    reader = SdkReader(api, Path("/tmp/config.json"), {}, clock=lambda: now[0])
    core = AE01Core(clock=lambda: now[0])
    try:
        core.observe(reader.poll())
        assert core.command() is not None
        now[0] += .3
        core.observe(reader.poll())
        assert core.command() is None
        # A raw frame may advance while the retarget worker remains stalled.
        devices["left"].stamps["raw"] += 1
        core.observe(reader.poll())
        assert core.command() is None
        devices["left"].stamps["robot"] += 1
        core.observe(reader.poll())
        assert core.command()[1:] == (True, False)
        infos[0].online = False
        core.observe(reader.poll())
        infos[0].online = True
        core.observe(reader.poll())
        assert core.command() is None  # Same cached source stamps after reconnection.
    finally:
        reader.close()
        reader.close()
    assert events == ["sdk.destroy"]
    assert all(dev.events[-1] == "stop" and "destroy" not in dev.events for dev in devices.values())


def test_sdk_duplicate_side_fault_and_bootstrap_gate():
    api, devices, infos, events = reader_fixture(lambda: 10.)
    reader = SdkReader(api, Path("/tmp/config.json"), {}, clock=lambda: 10.)
    core = AE01Core(clock=lambda: 10.)
    try:
        infos[0].diagnostics = {0x0940: [1]}
        infos[1].retarget_bootstrap_ready = False
        core.observe(reader.poll())
        assert core.command() is None
        infos[1].retarget_bootstrap_ready = True
        infos.append(copy.copy(infos[1]))
        packet = reader.poll()
        core.observe(packet)
        assert core.command() is None
        assert "多个" in packet["sides"]["right"]["error"]
    finally:
        reader.close()


def test_failed_sdk_initialize_is_destroyed():
    api, devices, infos, events = reader_fixture(lambda: 10., fail_init=True)
    with pytest.raises(RuntimeError, match="initialize"):
        SdkReader(api, Path("/tmp/config.json"), {})
    assert events == ["sdk.destroy"]


@pytest.mark.parametrize("transport", ["usb_serial", "ethernet_udp", "usb_vendor"])
def test_device_initialization_preserves_transport_retarget_and_resources(transport):
    api, devices, infos, events = reader_fixture(lambda: 10.)
    calls = []

    def initialize_device(config):
        options = json.loads(config)
        calls.append(options)
        if options.get("transport_link") != transport:
            return 1
        return 0

    for dev in devices.values():
        dev.init = initialize_device
    config = dict(transport_link=transport, casadi_common=dict(retarget_update_rate_hz=60),
                  hand_fk_config_root="/tmp/hand_fk", enable_wave=False)
    reader = SdkReader(api, Path("/tmp/config.json"), config, clock=lambda: 10.)
    try:
        packet = reader.poll()
        core = AE01Core(clock=lambda: 10.)
        core.observe(packet)
        assert core.command()[1:] == (True, True)
        assert len(calls) == 2 and all(call["enable_retarget"] for call in calls)
        assert all(call["transport_link"] == transport and call["casadi_common"] == config["casadi_common"]
                   and call["hand_fk_config_root"] == config["hand_fk_config_root"] for call in calls)
        assert all("enable_identity_query" not in call for call in calls)
    finally:
        reader.close()


def test_async_identity_connection_failure_is_retried_without_destroying_shared_handle():
    now = [10.]
    api, devices, infos, events = reader_fixture(lambda: now[0])
    attempts = []
    infos[0].sn = ""

    def initialize_left(_config):
        attempts.append(now[0])
        return 0 if infos[0].sn else 1

    devices["left"].init = initialize_left
    reader = SdkReader(api, Path("/tmp/config.json"), {}, clock=lambda: now[0])
    core = AE01Core(clock=lambda: now[0])
    try:
        core.observe(reader.poll())
        assert core.command()[1:] == (False, True)
        assert devices["left"].events == []  # No stop/destroy during async discovery.
        now[0] = 10.1
        core.observe(reader.poll())
        assert attempts == [10.] and not core.valid("left")
        # Native SDK completes identity asynchronously, including changing the initial empty SN.
        infos[0].sn = "LEFT-IDENTITY-READY"
        now[0] = 10.6
        core.observe(reader.poll())
        assert attempts == [10., 10.6] and core.valid("left")
        assert devices["left"].events == ["start"]
    finally:
        reader.close()
    assert events == ["sdk.destroy"]
    assert "destroy" not in devices["left"].events


@pytest.mark.parametrize("kind,count,known_layout,finite,valid", [
    ("raw", 22, True, True, True), ("raw", 32, True, True, True),
    ("raw", 32, False, True, False), ("raw", 33, True, True, False),
    ("raw", 22, True, False, False), ("robot", 32, True, True, False),
])
def test_raw_imu_extension_is_validated_and_never_becomes_a_robot_joint(kind, count, known_layout, finite, valid):
    api, devices, infos, events = reader_fixture(lambda: 10.)
    fetch = devices["left"].fetch_data

    def fetch_extended(category):
        code, frame = fetch(category)
        if category == kind:
            hand = getattr(frame, category)
            hand.joint.position = [.1]*22 + [float("nan")]*(count-22)
            if not finite:
                hand.joint.position[0] = float("nan")
            hand.joint.name = [f"joint_{i}" for i in range(22)] + list(RAW_IMU_FIELDS)
            if not known_layout:
                hand.joint.name[-1] = "unknown"
        return code, frame

    devices["left"].fetch_data = fetch_extended
    reader = SdkReader(api, Path("/tmp/config.json"), {}, clock=lambda: 10.)
    core = AE01Core(clock=lambda: 10.)
    try:
        core.observe(reader.poll())
        command = core.command()
        assert core.valid("left") is valid and core.valid("right")
        assert len(command[0]) == 44
        assert command[0][:22] == ([.1]*22 if valid else [0.]*22)
    finally:
        reader.close()


def test_existing_avatar_process_is_detected_without_killing_it(tmp_path):
    process = tmp_path/"12345"
    process.mkdir()
    (process/"cmdline").write_bytes(b"/opt/avatar-sdk/bin/avatar-backend\0--transport\0wireless\0")
    assert vendor_processes(tmp_path) == ["12345"]
    (process/"cmdline").write_bytes(b"/usr/bin/python3\0unrelated.py\0")
    assert vendor_processes(tmp_path) == []
