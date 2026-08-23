"""Episode recorder ROS node for the dual UR5 + SharpA contract."""
from __future__ import annotations
import json, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .episode_writer import EpisodeWriterCompat
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    from ur_msgs.msg import CameraSet, UrStateFrame, HandStateFrame, TauFrame, WrenchFrame, DeformationFrame
    from ur_common.camera_transport import SharedCameraReader
    ROS_AVAILABLE=True
except Exception: ROS_AVAILABLE=False; Node=object # type: ignore

FINGERS=("pinky","ring","middle","index","thumb")
TELEOP_HAND_SOURCE=2  # HandCommand.SOURCE_VIVE; shared source id for Manus teleop.

def _fk(q44):
    from ur_common.kinematics import UrCommonKinematics
    if not hasattr(_fk,"kin"): _fk.kin=UrCommonKinematics()
    p=_fk.kin.hand_fingertip_positions(q44)
    return p["left"].reshape(-1).tolist(), p["right"].reshape(-1).tolist()

def _vec(v,n):
    vals=list(v) if v is not None else []
    if len(vals)!=n: raise ValueError(f"expected vector length {n}, got {len(vals)}")
    result = [float(x) for x in vals]
    if np is not None and not np.isfinite(result).all():
        raise ValueError("vector contains non-finite values")
    return result

def _stamp_ns(msg, field="timestamp_ns"):
    x=getattr(msg,field,0); return int(x)

@dataclass
class _Latest:
    msg: Any=None; received: float=0.0

class RecorderCore:
    """ROS-independent synchronizer; callbacks only update latest samples."""
    def __init__(self, root, frequency=30.0, camera_max_age_s=.25, video_ffmpeg="ffmpeg"):
        self.writer=EpisodeWriterCompat(root,frequency,ffmpeg=video_ffmpeg); self.frequency=float(frequency); self.camera_max_age_s=float(camera_max_age_s)
        self.recording=False; self.started=time.monotonic(); self.stopped=0.0; self.frames=0; self.attempts=0; self.dropped=0; self.deformation_dropped=0; self.camera_dropped={name:0 for name in ("ego","left_wrist","right_wrist")}; self.last_error=""; self.quality_errors=[]; self.validation=None; self._last_video_timestamps={name:0 for name in self.camera_dropped}; self._lock=threading.RLock(); self.latest={k:_Latest() for k in ("camera","ur","hand","tau","wrench","deform")}
    def update(self,key,msg,received=None):
        if key == "camera":
            return self.update_camera(msg,received)
        with self._lock:self.latest[key]=_Latest(msg,time.time() if received is None else received)
    def update_camera(self,msg,received=None):
        received=time.time() if received is None else received
        with self._lock:
            self.latest["camera"]=_Latest(msg,received)
            if not self.recording:return True
            return self._record_camera_set(msg,received)
    def _record_camera_set(self,msg,received):
        accepted=True; timeline_row=max(0,self.attempts-1)
        for name in ("ego","left_wrist","right_wrist"):
            valid=bool(getattr(msg,f"{name}_valid",False)); timestamp=int(getattr(msg,f"{name}_timestamp_ns",0)); payload=getattr(msg,name,b"")
            previous=self._last_video_timestamps[name]
            if not valid or timestamp<=0 or timestamp==previous:continue
            if previous and timestamp<previous:
                self.camera_dropped[name]+=1; self._note_quality_error(f"{name} camera timestamp moved backwards"); accepted=False; continue
            if not self.writer.add_camera_frame(name,payload,timestamp,timeline_row,int(received*1e9)):
                self.camera_dropped[name]+=1; self._note_quality_error(f"{name} camera writer queue full"); accepted=False; continue
            self._last_video_timestamps[name]=timestamp
        return accepted
    def update_deformation(self,msg,received=None):
        received=time.time() if received is None else received
        with self._lock:
            self.latest["deform"]=_Latest(msg,received)
            if not self.recording:return True
            timeline_row=max(0,self.attempts-1)
            accepted=self.writer.add_deformation(msg,timeline_row,int(received*1e9))
            if not accepted:
                self.deformation_dropped+=1; self._note_quality_error("deformation writer queue full")
            return accepted
    def start(self):
        with self._lock:
            if self.recording:return False
            self.validation=None
            self.writer.start(); self.recording=True; self.started=time.monotonic(); self.stopped=0.0; self.frames=0; self.attempts=0; self.dropped=0; self.deformation_dropped=0; self.camera_dropped={name:0 for name in self.camera_dropped}; self.last_error=""; self.quality_errors=[]; self._last_video_timestamps={name:0 for name in self.camera_dropped}
            camera=self.latest["camera"]
            if camera.msg is not None:self._record_camera_set(camera.msg,time.time())
            return True
    def stop(self,save=True):
        with self._lock:self.recording=False; self.stopped=time.monotonic()
        result=self.writer.stop(save=save, dropped_frames=self.dropped,
                                quality_errors=self.quality_errors)
        with self._lock:
            self.validation=self.writer.last_validation if save else None
        return result
    def classify(self, category):
        with self._lock:
            if self.recording:
                raise RuntimeError("cannot classify while recording")
            destination=self.writer.classify(category)
            self.validation=self.writer.last_validation
            self.last_error=""
            return destination
    def drop_last_saved(self):
        with self._lock:
            if self.recording:
                raise RuntimeError("cannot drop a sample while recording")
            removed=self.writer.delete_last_saved()
            self.validation=None
            self.last_error=""
            return removed
    def _fresh(self,x,now): return x.msg is not None and now-x.received<=self.camera_max_age_s
    def _note_quality_error(self, message):
        self.last_error = str(message)
        if self.last_error not in self.quality_errors:
            self.quality_errors.append(self.last_error)
    @staticmethod
    def _finger_valid(msg, side):
        values = list(getattr(msg, f"{side}_valid", [])) if msg is not None else []
        return [bool(value) for value in values] if len(values) == 5 else [False] * 5
    def _tactile_quality_errors(self):
        errors = []
        for key, label in (("wrench", "wrench"), ("deform", "deformation")):
            msg = self.latest[key].msg
            missing = []
            for side in ("left", "right"):
                valid = self._finger_valid(msg, side)
                missing.extend(f"{side}_{finger}" for finger, ok in zip(FINGERS, valid) if not ok)
            if missing:
                errors.append(f"SharpA {label} invalid: {', '.join(missing)}")
        return errors
    def sample(self, now=None):
        now=time.time() if now is None else now
        with self._lock:
            if not self.recording:return False
            required=("camera","ur","hand","tau","wrench","deform")
            missing=[k for k in required if not self._fresh(self.latest[k],now)]
            self.writer.add_raw_row(self._build_raw_row(now))
            timeline_index = self.attempts
            self.attempts += 1
            if missing:
                self.dropped+=1; self.last_error="missing/stale modalities: "+", ".join(missing); self.quality_errors.append(self.last_error); return False
            for error in self._tactile_quality_errors(): self._note_quality_error(error)
            try:item=self._build_item(now)
            except Exception as exc:self.dropped+=1; self.last_error=str(exc); self.quality_errors.append(self.last_error); return False
            item["metadata"]["timeline_index"] = timeline_index
            if not self.writer.add_item(item): self.dropped+=1; self.last_error="writer queue full"; self.quality_errors.append(self.last_error); return False
            self.frames+=1; return True
    def _build_raw_row(self, now):
        source_names=("camera","ur","hand","tau","wrench","deform")
        source_fresh={key:self._fresh(self.latest[key],now) for key in source_names}
        source_valid=dict(source_fresh)
        cam=self.latest["camera"].msg; ur=self.latest["ur"].msg; hand=self.latest["hand"].msg; tau=self.latest["tau"].msg; wrench=self.latest["wrench"].msg; deform=self.latest["deform"].msg
        if source_valid["camera"]: source_valid["camera"]=bool(cam.ego_valid and cam.left_wrist_valid and cam.right_wrist_valid)
        if source_valid["ur"]: source_valid["ur"]=bool(getattr(ur,"valid",True))
        if source_valid["hand"]: source_valid["hand"]=bool(hand.left_valid and hand.right_valid)
        if source_valid["tau"]: source_valid["tau"]=bool(tau.left_valid and tau.right_valid)
        wrench_valid=[self._finger_valid(wrench, side) if source_fresh["wrench"] else [False]*5 for side in ("left","right")]
        deformation_valid=[self._finger_valid(deform, side) if source_fresh["deform"] else [False]*5 for side in ("left","right")]
        source_valid["wrench"]=bool(source_fresh["wrench"] and all(v for side in wrench_valid for v in side))
        source_valid["deform"]=bool(source_fresh["deform"] and all(v for side in deformation_valid for v in side) and len(deform.left)==5*240*240 and len(deform.right)==5*240*240)
        def values(msg, field, size, valid):
            if not valid or msg is None:return [float("nan")]*size
            try:return _vec(getattr(msg,field),size)
            except Exception:return [float("nan")]*size
        ages=[float("inf") if self.latest[key].msg is None else max(0.0,(now-self.latest[key].received)*1000.0) for key in source_names]
        return {"time_unix_ns":int(now*1e9),"source_names":source_names,"source_valid":[source_valid[k] for k in source_names],"source_age_ms":ages,
                "source_timestamp_ns":[
                    (max(int(getattr(cam, f"{name}_timestamp_ns", 0)) for name in ("ego", "left_wrist", "right_wrist")) if cam is not None else 0),
                    *[0 if msg is None else _stamp_ns(msg) for msg in (ur, hand, tau, wrench, deform)],
                ],
                "ur_q":[*values(ur,"left_joint",6,source_valid["ur"]),*values(ur,"right_joint",6,source_valid["ur"])],
                "ur_tcp":[values(ur,"left_tcp_pose6",6,source_valid["ur"] and hasattr(ur,"left_tcp_pose6")),values(ur,"right_tcp_pose6",6,source_valid["ur"] and hasattr(ur,"right_tcp_pose6"))],
                "ur_wrist":[values(ur,"left_eef",9,source_valid["ur"]),values(ur,"right_eef",9,source_valid["ur"])],
                "hand_q":values(hand,"q",44,source_valid["hand"]),
                "hand_q_cmd":values(hand,"q_cmd",44,source_valid["hand"]),
                "hand_q_cmd_valid":bool(source_valid["hand"] and getattr(hand,"q_cmd_valid",False) and int(getattr(hand,"q_cmd_source",0))==TELEOP_HAND_SOURCE),
                "hand_q_cmd_timestamp_ns":int(getattr(hand,"q_cmd_timestamp_ns",0)) if hand is not None else 0,
                "hand_q_cmd_source":int(getattr(hand,"q_cmd_source",0)) if hand is not None else 0,
                "hand_tau":[*values(tau,"left",22,source_valid["tau"]),*values(tau,"right",22,source_valid["tau"])],
                "wrench":[[values(wrench,side,30,source_fresh["wrench"])[i*6:(i+1)*6] for i in range(5)] for side in ("left","right")],
                "wrench_valid":wrench_valid,"deformation_valid":deformation_valid,
                "camera_timestamp_ns":[int(getattr(cam,f"{name}_timestamp_ns",0)) if source_valid["camera"] else 0 for name in ("ego","left_wrist","right_wrist")]}
    def _build_item(self,now):
        cam=self.latest["camera"].msg; ur=self.latest["ur"].msg; hand=self.latest["hand"].msg; tau=self.latest["tau"].msg; wrench=self.latest["wrench"].msg; deform=self.latest["deform"].msg
        if str(getattr(cam,"encoding","jpeg")) != "jpeg": raise ValueError("camera encoding must be jpeg")
        if not bool(getattr(ur, "valid", True)): raise ValueError("UR state invalid")
        if not bool(getattr(hand,"left_valid",True)) or not bool(getattr(hand,"right_valid",True)): raise ValueError("SharpA hand state invalid")
        if not bool(getattr(hand,"q_cmd_valid",False)): raise ValueError("SharpA hand action is missing or stale")
        if int(getattr(hand,"q_cmd_source",0)) != TELEOP_HAND_SOURCE: raise ValueError("SharpA hand action is not from Manus teleop")
        q_cmd_timestamp_ns=int(getattr(hand,"q_cmd_timestamp_ns",0))
        if q_cmd_timestamp_ns <= 0: raise ValueError("SharpA hand action timestamp is invalid")
        if not bool(getattr(tau,"left_valid",True)) or not bool(getattr(tau,"right_valid",True)): raise ValueError("SharpA tau invalid")
        if len(getattr(deform,"left",[])) != 5*240*240 or len(getattr(deform,"right",[])) != 5*240*240: raise ValueError("SharpA deformation shape invalid")
        q44=_vec(hand.q,44); q_cmd44=_vec(hand.q_cmd,44); ltip,rtip=_fk(q44)
        colors={"color_0":"cameras/ego/rgb.mp4","color_1":"cameras/left_wrist/rgb.mp4","color_2":"cameras/right_wrist/rgb.mp4"}
        if not (cam.ego_valid and cam.left_wrist_valid and cam.right_wrist_valid): raise ValueError("camera view flagged invalid")
        states={"left_arm":{"qpos":_vec(ur.left_joint,6)},"right_arm":{"qpos":_vec(ur.right_joint,6)},"left_arm_pose":{"pose":_vec(ur.left_eef,9)},"right_arm_pose":{"pose":_vec(ur.right_eef,9)},"left_hand":{"qpos":q44[:22]},"right_hand":{"qpos":q44[22:]},"left_hand_positions":{"positions":ltip},"right_hand_positions":{"positions":rtip},"left_hand_torques":{"torques":_vec(tau.left,22)},"right_hand_torques":{"torques":_vec(tau.right,22)},"tactile":{"left_tactile":self._wrench_dict(wrench.left),"right_tactile":self._wrench_dict(wrench.right)}}
        if hasattr(ur, "left_tcp_pose6") and hasattr(ur, "right_tcp_pose6"):
            states["left_arm"]["tcp_pose6"] = _vec(ur.left_tcp_pose6, 6)
            states["right_arm"]["tcp_pose6"] = _vec(ur.right_tcp_pose6, 6)
        # The fresh received Manus hand target is the teleop action, even while
        # SharpA is paused. Vive arm commands are intentionally not recorded.
        actions={"left_hand":{"qpos":q_cmd44[:22]},"right_hand":{"qpos":q_cmd44[22:]}}
        stamps={"frame_start":now,"wall_clock":now,"wrist":now,"head_image":int(cam.ego_timestamp_ns)/1e9,"left_image":int(cam.left_wrist_timestamp_ns)/1e9,"right_image":int(cam.right_wrist_timestamp_ns)/1e9,"arm":int(getattr(ur,"timestamp_ns",0))/1e9,"hand_state":int(getattr(hand,"timestamp_ns",0))/1e9,"hand_action":q_cmd_timestamp_ns/1e9,"tactile":int(getattr(tau,"timestamp_ns",0))/1e9,"frame_end":time.time()}
        validity={kind:{side:self._finger_valid(msg,side) for side in ("left","right")} for kind,msg in (("wrench",wrench),("deformation",deform))}
        return {"colors":colors,"depths":{},"states":states,"actions":actions,"tactiles":None,"audios":None,"tactile_images":None,"tactile_images_raw":{},"tactile_images_deform":{},"metadata":{"frame_count":self.frames,"stamps":stamps,"hand_action_source":int(hand.q_cmd_source),"tactile_valid":validity}}
    @staticmethod
    def _wrench_dict(values):
        vals=_vec(values,30); return {finger:vals[i*6:(i+1)*6] for i,finger in enumerate(FINGERS)}
    def status(self):
        end=time.monotonic() if self.recording or not self.stopped else self.stopped; elapsed=max(0.0,end-self.started); return {"recording":self.recording,"episode_id":self.writer.episode_id,"frame_count":self.frames,"timeline_rows":self.attempts,"duration_s":round(elapsed,3),"achieved_rate_hz":round(self.frames/max(1e-9,elapsed),3),"dropped_frames":self.dropped,"deformation_dropped":self.deformation_dropped,"camera_frames":self.writer.camera_frame_counts(),"camera_dropped":dict(self.camera_dropped),"last_error":self.last_error,"recording_root":str(self.writer.root),"validation":self.validation}

if ROS_AVAILABLE:
    class RecordingNode(Node):
        def __init__(self):
            super().__init__("recording_node"); self.declare_parameter("recording_root","./episodes"); self.declare_parameter("frequency",30.0); self.declare_parameter("camera_max_age_s",.25); self.declare_parameter("shared_memory_prefix","ur_sharpa_camera_v1"); self.declare_parameter("video_ffmpeg","ffmpeg")
            frequency=float(self.get_parameter("frequency").value); self.core=RecorderCore(str(self.get_parameter("recording_root").value),frequency,float(self.get_parameter("camera_max_age_s").value),video_ffmpeg=str(self.get_parameter("video_ffmpeg").value)); q=QoSProfile(depth=10); cq=QoSProfile(depth=1,history=HistoryPolicy.KEEP_LAST,reliability=ReliabilityPolicy.BEST_EFFORT); self.status_pub=self.create_publisher(String,"/control/record_status",q)
            self.create_subscription(String,"/control/record_command",self._command,q); self.create_subscription(CameraSet,"/camera/set",self._camera,cq); self.create_subscription(UrStateFrame,"/ur/state",lambda m:self.core.update("ur",m),q); self.create_subscription(HandStateFrame,"/sharpa/state",lambda m:self.core.update("hand",m),q); self.create_subscription(TauFrame,"/sharpa/tau",lambda m:self.core.update("tau",m),q); self.create_subscription(WrenchFrame,"/sharpa/wrench",lambda m:self.core.update("wrench",m),q); self.create_subscription(DeformationFrame,"/sharpa/deformation",self.core.update_deformation,cq); self.create_timer(1.0,self._publish)
            self.create_subscription(String,"/control/record_config",self._config,q)
            self.create_subscription(String,"/control/record_label",self._label,q)
            self.camera_reader=SharedCameraReader(str(self.get_parameter("shared_memory_prefix").value)); self._shared_camera_seen=False; self._observed_camera_timestamps=None; self.create_timer(1.0/max(30.0,frequency*4.0),self._poll_camera); self.create_timer(1.0/frequency,self.core.sample)
        def _command(self,msg):
            if msg.data.strip()=="start": self.core.start()
            elif msg.data.strip()=="stop": self.core.stop(True)
            elif msg.data.strip()=="discard": self.core.stop(False)
        def _camera(self,msg):
            if not self._shared_camera_seen:self.core.update_camera(msg)
        def _poll_camera(self):
            msg=self.camera_reader.read()
            if msg is None:return
            self._shared_camera_seen=True
            timestamps=(msg.ego_timestamp_ns,msg.left_wrist_timestamp_ns,msg.right_wrist_timestamp_ns)
            if timestamps==self._observed_camera_timestamps:return
            self._observed_camera_timestamps=timestamps; self.core.update_camera(msg)
        def _config(self,msg):
            if self.core.recording:return
            try:
                root=json.loads(msg.data).get("recording_root")
                if root:self.core.writer.root=Path(str(root)).expanduser()
            except Exception as exc:self.get_logger().warning(f"invalid record config: {exc}")
        def _label(self,msg):
            try:
                if msg.data.strip().lower() == "drop":
                    removed=self.core.drop_last_saved()
                    self.get_logger().warning(
                        f"sample permanently deleted: {removed}"
                    )
                else:
                    destination=self.core.classify(msg.data)
                    self.get_logger().info(f"sample classified: {destination}")
            except Exception as exc:
                self.core.last_error=f"sample classification/drop failed: {exc}"
                self.get_logger().warning(self.core.last_error)
        def _publish(self): m=String(); m.data=json.dumps(self.core.status(),separators=(",",":")); self.status_pub.publish(m)
        def destroy_node(self): self.camera_reader.close(); self.core.stop(True); self.core.writer.close(); return super().destroy_node()
else:
    class RecordingNode: # pragma: no cover
        def __init__(self,*_a,**_kw): raise RuntimeError("rclpy is unavailable")

def main(args=None):
    if not ROS_AVAILABLE: raise RuntimeError("rclpy is unavailable")
    rclpy.init(args=args); node=RecordingNode()
    try:rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, RCLError):pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
