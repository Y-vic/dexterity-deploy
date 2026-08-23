import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from recording_node.recording_node import RecorderCore
from recording_node.episode_writer import MIN_VALID_RATE_HZ, _validation_rate


def jpeg_bytes(color=(32, 64, 96)):
    output=BytesIO()
    Image.new("RGB",(32,24),color).save(output,format="JPEG",quality=90)
    return output.getvalue()


def test_validation_rate_threshold_is_25_hz():
    assert MIN_VALID_RATE_HZ == 25.0
    assert _validation_rate(25, 1.0) >= MIN_VALID_RATE_HZ
    assert _validation_rate(24, 1.0) < MIN_VALID_RATE_HZ

def test_fingertip_fk_shape_when_kinematics_available():
    sys.path.insert(0, str(Path(__file__).parents[2] / "ur_common"))
    pytest.importorskip("pinocchio")
    from ur_common.kinematics import UrCommonKinematics
    poses=UrCommonKinematics().hand_fingertip_positions([0.0]*44)
    assert poses["left"].shape==(5,3) and poses["right"].shape==(5,3)

def msgset():
    jpg=jpeg_bytes()
    cam=SimpleNamespace(ego=jpg,left_wrist=jpg,right_wrist=jpg,ego_valid=True,left_wrist_valid=True,right_wrist_valid=True,ego_timestamp_ns=1_000_000_000,left_wrist_timestamp_ns=1_000_000_000,right_wrist_timestamp_ns=1_000_000_000,encoding="jpeg")
    ur=SimpleNamespace(left_joint=[0]*6,right_joint=[0]*6,left_eef=[0]*9,right_eef=[0]*9,left_tcp_pose6=[0]*6,right_tcp_pose6=[0]*6,timestamp_ns=1_000_000_001)
    hand=SimpleNamespace(q=[0]*44,q_cmd=[0.25]*22+[-0.25]*22,q_cmd_valid=True,q_cmd_timestamp_ns=999_000_000,q_cmd_source=2,timestamp_ns=1_000_000_000,left_valid=True,right_valid=True)
    tau=SimpleNamespace(left=[0]*22,right=[0]*22,left_valid=True,right_valid=True,timestamp_ns=1_000_000_000)
    wrench=SimpleNamespace(left=[0]*30,right=[0]*30,left_valid=[True]*5,right_valid=[True]*5,timestamp_ns=1_000_000_000)
    deform=SimpleNamespace(left=[0]*(5*240*240),right=[0]*(5*240*240),left_valid=[True]*5,right_valid=[True]*5,timestamp_ns=1_000_000_000)
    return {"camera":cam,"ur":ur,"hand":hand,"tau":tau,"wrench":wrench,"deform":deform}

def start_with_deformation(c):
    assert c.start()
    assert c.update_deformation(c.latest["deform"].msg,received=1_000_000_000)

def test_all_or_nothing_discard(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    # Avoid optional pinocchio in this unit test while still testing shape/writer.
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    for k,v in msgset().items(): c.update(k,v,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000)
    out=c.stop(save=False); assert out is None; assert not list(Path(tmp_path).glob("sample_*"))

def test_output_schema_and_three_color_mapping(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    for k,v in msgset().items(): c.update(k,v,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000); out=c.stop(save=True); assert out
    d=json.loads((out/"data.json").read_text()); assert set(d)=={"info","data"}; f=d["data"][0]
    assert set(f)=={"idx","colors","depths","states","actions","tactiles","audios","tactile_images","tactile_images_raw","tactile_images_deform","metadata"}
    assert set(f["colors"])=={"color_0","color_1","color_2"}; assert set(f["metadata"]["stamps"])=={"frame_start","wall_clock","wrist","head_image","left_image","right_image","arm","hand_state","hand_action","tactile","frame_end"}
    assert f["colors"] == {
        "color_0":"cameras/ego/rgb.mp4",
        "color_1":"cameras/left_wrist/rgb.mp4",
        "color_2":"cameras/right_wrist/rgb.mp4",
    }
    assert f["actions"] == {
        "left_hand":{"qpos":[0.25]*22},
        "right_hand":{"qpos":[-0.25]*22},
    }
    assert f["metadata"]["stamps"]["hand_action"] == pytest.approx(0.999)
    assert f["metadata"]["hand_action_source"] == 2
    for name in ("ego","left_wrist","right_wrist"):
        video=out/"cameras"/name/"rgb.mp4"
        assert video.is_file() and video.stat().st_size > 0
        index=np.load(out/"cameras"/name/"index.npz")
        assert index["frame_index"].tolist() == [0]
        assert index["source_timestamp_ns"].tolist() == [1_000_000_000]
    assert not list(out.rglob("*.jpg"))
    probe=json.loads(subprocess.run(
        ["ffprobe","-v","error","-show_entries","stream=codec_name,nb_frames","-of","json",str(out/"cameras"/"ego"/"rgb.mp4")],
        check=True,capture_output=True,text=True,
    ).stdout)
    assert probe["streams"][0]["codec_name"] == "mjpeg"
    assert probe["streams"][0]["nb_frames"] == "1"
    validation=json.loads((out/"validation.json").read_text())
    assert [item["key"] for item in validation["items"]] == [
        "ego_camera", "left_wrist_camera", "right_wrist_camera",
        "arm_state", "hand_state", "hand_action", "wrench", "deform",
    ]
    assert [item["label"] for item in validation["items"][:4]] == [
        "Ego Camera", "Left Wrist Camera", "Right Wrist Camera", "Arm State",
    ]
    assert validation["ok"] is True
    assert all(item["frames"] == 1 and item["missing"] == 0 for item in validation["items"])
    assert c.status()["validation"] == validation


def test_starting_next_episode_clears_previous_validation(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    for key,value in msgset().items(): c.update(key,value,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000)
    assert c.stop(save=True)
    assert c.status()["validation"] is not None
    assert c.start()
    assert c.status()["validation"] is None
    c.stop(save=False)

def test_raw_contract_preserves_source_timestamps_and_ur_poses(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    messages=msgset()
    for k,v in messages.items(): c.update(k,v,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000); out=c.stop(save=True)
    ur=np.load(out/"ur"/"physical_12.npz")
    assert ur["q"].shape == (1,12)
    assert ur["raw_tcp_pose6"].shape == (1,2,6)
    assert ur["canonical_wrist_pose9"].shape == (1,2,9)
    assert ur["timestamp_ns"].tolist() == [messages["ur"].timestamp_ns]
    sharpa=np.load(out/"sharpa"/"joints_44.npz")
    assert sharpa["q_timestamp_ns"].tolist() == [messages["hand"].timestamp_ns]
    assert sharpa["tau_timestamp_ns"].tolist() == [messages["tau"].timestamp_ns]
    np.testing.assert_allclose(sharpa["q_cmd"][0,:22],0.25)
    np.testing.assert_allclose(sharpa["q_cmd"][0,22:],-0.25)
    assert sharpa["q_cmd_valid"].tolist() == [True]
    assert sharpa["q_cmd_timestamp_ns"].tolist() == [999_000_000]
    assert sharpa["q_cmd_source"].tolist() == [2]

def test_missing_hand_action_is_not_fabricated_from_measured_state(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    messages=msgset(); messages["hand"].q_cmd_valid=False
    for key,value in messages.items(): c.update(key,value,received=1_000_000_000)
    start_with_deformation(c)
    assert not c.sample(now=1_000_000_000)
    assert "hand action is missing or stale" in c.last_error
    assert c.frames == 0 and c.dropped == 1

def test_repeated_camera_source_timestamp_is_reused_without_duplicate_video_frame(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    for k,v in msgset().items(): c.update(k,v,received=1_000_000_000)
    start_with_deformation(c)
    assert c.sample(now=1_000_000_000)
    assert c.sample(now=1_000_000_000)
    assert c.frames == 2
    assert c.dropped == 0
    out=c.stop(save=True)
    assert out
    for name in ("ego","left_wrist","right_wrist"):
        index=np.load(out/"cameras"/name/"index.npz")
        assert index["frame_index"].tolist() == [0]

def test_camera_views_are_recorded_independently(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    messages=msgset()
    for key,value in messages.items(): c.update(key,value,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000)
    camera=messages["camera"]
    camera.ego_timestamp_ns=1_033_000_000
    camera.right_wrist_timestamp_ns=1_040_000_000
    c.update_camera(camera,received=1_000_000_000.04)
    assert c.sample(now=1_000_000_000.04)
    out=c.stop(save=True); assert out
    expected={"ego":2,"left_wrist":1,"right_wrist":2}
    for name,count in expected.items():
        index=np.load(out/"cameras"/name/"index.npz")
        assert index["frame_index"].tolist() == list(range(count))
        meta=json.loads((out/"cameras"/name/"rgb_meta.json").read_text())
        assert meta["frame_count"] == count
        assert meta["probe"]["frame_count"] == count

def test_partial_tactile_frame_is_saved_and_reports_finger_percentages(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    messages=msgset()
    messages["wrench"].left_valid[1]=False
    messages["deform"].right_valid[4]=False
    for k,v in messages.items(): c.update(k,v,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000)
    assert c.frames == 1 and c.dropped == 0
    out=c.stop(save=True); assert out
    force=np.load(out/"sharpa"/"tactile"/"force6d.npz")
    deform=np.load(out/"sharpa"/"tactile"/"deform_index.npz")
    assert force["valid"].shape == (1,2,5) and not force["valid"][0,0,1]
    assert deform["valid"].shape == (1,10) and not deform["valid"][0,4]
    validation=c.status()["validation"]
    assert validation and validation["ok"] is False
    by_key={item["key"]:item for item in validation["items"]}
    assert "左无名指 100.00%" in by_key["wrench"]["details"]
    assert "右拇指 100.00%" in by_key["deform"]["details"]
    assert by_key["wrench"]["finger_missing"] == [{
        "finger":"左无名指", "missing_frames":1, "total_frames":1,
        "ratio_percent":100.0,
    }]


def test_classification_moves_sample_and_next_id_stays_global(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    for key,value in msgset().items(): c.update(key,value,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000)
    root_sample=c.stop(save=True); assert root_sample.parent == tmp_path
    destination=c.classify("success")
    assert destination == tmp_path/"success"/"sample_0000"
    assert destination.is_dir() and not root_sample.exists()
    assert c.status()["validation"]["category"] == "success"
    assert c.start()
    assert c.writer.episode_id == 1
    c.stop(save=False)


def test_drop_permanently_deletes_only_last_saved_sample(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    unrelated=tmp_path/"keep.txt"; unrelated.write_text("keep")
    for key,value in msgset().items(): c.update(key,value,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000)
    sample=c.stop(save=True); assert sample and sample.is_dir()
    removed=c.drop_last_saved()
    assert removed == sample.resolve()
    assert not sample.exists()
    assert unrelated.read_text() == "keep"
    assert c.status()["validation"] is None

def test_deformation_uses_pnd_raw_stream_and_plane_order(tmp_path):
    c=RecorderCore(tmp_path,frequency=30)
    import recording_node.recording_node as mod
    mod._fk=lambda q: ([0.0]*15,[0.0]*15)
    messages=msgset()
    messages["deform"].left=[11]*(5*240*240)
    messages["deform"].right=[22]*(5*240*240)
    for key,value in messages.items(): c.update(key,value,received=1_000_000_000)
    start_with_deformation(c); assert c.sample(now=1_000_000_000); out=c.stop(save=True)
    raw=subprocess.run(
        ["zstd", "-q", "-d", "-c", str(out/"sharpa"/"tactile"/"deform_u8.zst")],
        check=True,
        capture_output=True,
    ).stdout
    deformation=np.frombuffer(raw,dtype=np.uint8).reshape(1,10,240,240)
    assert np.all(deformation[0,:5] == 22)
    assert np.all(deformation[0,5:] == 11)
    index=np.load(out/"sharpa"/"tactile"/"deform_index.npz")
    assert index["raw_frame_index"].tolist() == [0]
    assert index["valid"].shape == (1,10)
    assert not list(out.glob("tactiles_deform/*.jpg"))
    assert not list((out/"sharpa"/"tactile").glob("*.npy"))
