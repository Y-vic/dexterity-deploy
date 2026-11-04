import sys, io, numpy as np
sys.path.insert(0,'/home/yns/deploy/Deploy-v2/interface')
sys.path.insert(0,'/home/yns/deploy/Deploy-v2/.work/deploy_v5')
from PIL import Image
from sharpa_interface.server.transport import SharpaV3PolicyClient
from sharpa_interface.server.action import parse_policy_action
from sharpa_interface.server.metadata import validate_server_metadata
from test_contract import make_observation

buf = io.BytesIO()
Image.fromarray(np.random.randint(0,255,(256,256,3),dtype=np.uint8)).save(buf, format="JPEG")
jpeg = buf.getvalue()
print("jpeg bytes:", len(jpeg))

import os
PORT = int(os.environ.get("SHARPA_PROBE_PORT", "5500"))
c = SharpaV3PolicyClient(url=f"ws://127.0.0.1:{PORT}/infer", ssh_host="orca",
                         ssh_remote_host="127.0.0.1", ssh_remote_port=PORT, timeout_s=180.0)
try:
    md = validate_server_metadata(c.metadata)
    print("metadata:", md["policy_family"], "| ckpt:", md["checkpoint_id"])
    sid = c.reset("real-gpu")["session_id"]
    print("session:", sid)
    for rid in range(3):
        obs = make_observation(sid, rid)
        obs["image"]["ego_cam"]["current"] = {"data": jpeg}
        # make_observation already carries a 12-frame history advancing 2 mm per
        # frame, so the state rows span ~24 mm -- the training magnitude.
        r = c.infer(obs)
        if r.payload.get("schema") == "sharpa_policy_error.v1":
            print("ERROR:", r.payload["error"]); break
        p = parse_policy_action(r.payload, expected_session_id=sid, expected_request_id=rid)
        eef = r.payload["action"]["left_wrist"]["eef"]
        hand = r.payload["action"]["hand_joint"]["left"]
        d = np.linalg.norm(eef[-1][:3] - obs["state"]["current"]["left_wrist"]["eef"][:3])
        print(f"INFER {rid}: {r.latency_s:.2f}s eef{eef.shape} hand{hand.shape} "
              f"|dpos 1s|={d*1000:.1f}mm finite={np.all(np.isfinite(eef))}")
finally:
    c.close()
