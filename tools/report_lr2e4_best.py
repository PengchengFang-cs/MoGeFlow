"""Best-checkpoint summary for the lr2e-4@1000ep arms (one line = one checkpoint)."""
import json

RUNS = [
    ("hml3d_x0L1_ep1000_lr2e4_20260814", "L1"),
    ("hml3d_x0L7_ep1000_lr2e4_20260814", "L7"),
    ("hml3d_x0L2_ep1000_lr2e4_20260814", "L2"),
]

for run, lab in RUNS:
    rows = [json.loads(l) for l in open(f"checkpoints/t2m/{run}/logs/full_eval.jsonl")]
    bf = min(rows, key=lambda x: x["aggregate"]["fid"])
    bt = max(rows, key=lambda x: x["aggregate"]["top3"])
    la = rows[-1]

    def s(x):
        a = x["aggregate"]
        return (f"ep{x['epoch']:>3}  FID {a['fid']:.4f}  R@3 {a['top3']:.4f}  "
                f"R@1 {a['top1']:.4f}  MMD {a['matching_score']:.4f}")

    print(f"== {lab}  ({len(rows)} evals, latest ep{la['epoch']}: "
          f"{la['aggregate']['fid']:.4f}/{la['aggregate']['top3']:.4f})")
    print(f"   best_fid : {s(bf)}")
    print(f"   best_top3: {s(bt)}")
