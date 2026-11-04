"""Clone the urscratch 16-GPU run into a `urhandcmd` variant.

Four files are cloned. The rename is a global `urscratch` -> `urhandcmd`, which is
correct for every occurrence: exp_name, the `defaults: - train:` pointer, the
normalizer dir, the three default `exp=` strings, and every self-reference in usage
text. It also turns `smoke_shared_mot_action_v5_urscratch` into
`smoke_shared_mot_action_v5_urhandcmd`, keeping the smoke_ prefix -- which is what we
want. The `cmp -s`/md5 reference set (ur_unscrew, urpn, head100h, mixed_pernorm)
contains no `urscratch`, so it is left alone.

Value changes, matched on STRIPPED line content so the original indentation is kept
and the comment that quotes `left_hand_joints: 0.5` in prose is not touched:
  left/right_hand_joints  0.5  -> 2.0     (hand share of the weight vector 45% -> 77%)
  action_loss_weight      0.15 -> 0.064   (holds total action-loss magnitude: 0.15*49/115)

checknorm's five hardcoded scale constants were measured against the OLD targets.
The three wrist dims (2, 3, 9) stay valid because the wrist action target is
byte-identical under this change. The two hand dims and the median hand scale are
NOT valid any more, so they are removed rather than left to silently mis-judge a
correct refit; the cloned check prints the measured hand median for step 4 to read.
"""
import os
import shutil

CONF = "/share/project/zjk/orca/examples/qwen3_5_gr00t/conf"
ROOT = "/share/project/zjk/orca"
OLD = "urscratch"
NEW = "urhandcmd"

FILES = [
    (f"{CONF}/train_mot_action_v5_urscratch_16gpu.yaml", f"{CONF}/train_mot_action_v5_urhandcmd_16gpu.yaml"),
    (f"{CONF}/train/qwen3_5_gr00t_mot_action_v5_urscratch_16gpu.yaml", f"{CONF}/train/qwen3_5_gr00t_mot_action_v5_urhandcmd_16gpu.yaml"),
    (f"{CONF}/train_mot_action_v5_urscratch_smoke2gpu.yaml", f"{CONF}/train_mot_action_v5_urhandcmd_smoke2gpu.yaml"),
    (f"{CONF}/train/qwen3_5_gr00t_mot_action_v5_urscratch_smoke2gpu.yaml", f"{CONF}/train/qwen3_5_gr00t_mot_action_v5_urhandcmd_smoke2gpu.yaml"),
    (f"{ROOT}/run_mot_action_v5_urscratch_16gpu.sh", f"{ROOT}/run_mot_action_v5_urhandcmd_16gpu.sh"),
]

# stripped-content -> replacement stripped-content
VALUE_EDITS = {
    "left_hand_joints: 0.5": "left_hand_joints: 2.0",
    "right_hand_joints: 0.5": "right_hand_joints: 2.0",
    "action_loss_weight: 0.15": "action_loss_weight: 0.064",
    '"L pinky_CMC (dim 29)": (29, 0.05038),': "# hand dims removed: the hand action target changed (resampled Manus command),",
    '"R pinky_CMC (dim 51)": (51, 0.02854),': "# so the old hand-dim constants would mis-judge a correct refit. Wrist dims stand.",
}

MEDIAN_OLD = 'print(f"  median hand-dim scale {statistics.median(hand):.5f}  (unscrew_cap: 0.09240)")'
MEDIAN_NEW = 'print(f"  median hand-dim scale {statistics.median(hand):.5f}  (RE-MEASURE: old achieved-q fit was 0.09240)")'


def leading(line):
    return line[: len(line) - len(line.lstrip())]


def edit_line(line):
    key = line.strip()
    if key in VALUE_EDITS:
        return leading(line) + VALUE_EDITS[key]
    if key == MEDIAN_OLD:
        return leading(line) + MEDIAN_NEW
    return line


def clone(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    text = open(src).read()
    text = text.replace(OLD, NEW)
    out = [edit_line(l) for l in text.split("\n")]
    open(dst, "w").write("\n".join(out))
    shutil.copymode(src, dst)
    return dst


def audit(path):
    text = open(path).read()
    hits = {
 "urscratch_left": text.count(OLD),
        "hand_joints_2.0": text.count("hand_joints: 2.0"),
        "hand_joints_0.5": len([l for l in text.split("\n") if l.strip() in ("left_hand_joints: 0.5", "right_hand_joints: 0.5")]),
        "alw_0.064": text.count("action_loss_weight: 0.064"),
        "alw_0.15": len([l for l in text.split("\n") if l.strip() == "action_loss_weight: 0.15"]),
    }
    return hits


made = [clone(s, d) for s, d in FILES]
print("CLONED:")
for p in made:
    print("  ", p)
    print("      ", audit(p))

# The normalizer dir must not exist yet -- precheck asserts this, but say it here too.
nd = f"{ROOT}/outputs/shared_mot_action_v5_{NEW}"
sd = f"{ROOT}/outputs/smoke_shared_mot_action_v5_{NEW}"
print()
print("normalizer dirs (must be absent/empty before first launch):")
print("  ", nd, "exists:", os.path.exists(nd))
print("  ", sd, "exists:", os.path.exists(sd))
