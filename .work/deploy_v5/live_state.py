"""The state the LIVE robot produces, in normalized units, for comparison."""
import json,sys,urllib.request
import numpy as np
sys.path.insert(0,"/home/yns/deploy/Deploy-v2/.work/deploy_v5")
import repr_bridge as rb

d=json.load(urllib.request.urlopen("http://127.0.0.1:8088/api/status",timeout=20))
ur=d["ur"]; sh=d["sharpa"]
anchor={"left":np.array(ur["left_eef"],dtype=np.float32),
        "right":np.array(ur["right_eef"],dtype=np.float32)}
hands={"left":np.array(sh["left_hand"],dtype=np.float32),
       "right":np.array(sh["right_hand"],dtype=np.float32)}
# One row at the current pose against itself is zero by construction, so this
# only reports the HAND block, which is absolute and therefore meaningful from a
# single frame. The wrist blocks need the history the server reads per request.
row=rb.build_state_vector(anchor,anchor,hands)
nm=json.load(open("/tmp/norm.json")) if False else None
print("hand block RAW rad: mean|.|=%.3f max|.|=%.3f"%(np.abs(row[12:34]).mean(),np.abs(row[12:34]).max()))
print("left_hand  raw:",np.round(hands["left"],3).tolist())
