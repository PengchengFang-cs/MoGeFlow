"""KIT x0 runs: best checkpoints (one line = one checkpoint) + gate saves."""
import json, os
RUNS = [("kit_x0L1_ep1000_20260818", "KIT x0-L1 1000ep"),
        ("kit_x0L1_e1_nopool_20260816", "KIT x0-L1 600ep")]
for run, lab in RUNS:
    p = f"checkpoints/kit/{run}/logs/full_eval.jsonl"
    if not os.path.exists(p):
        print(f"== {lab}: 无日志"); continue
    rows = [json.loads(l) for l in open(p)]
    bf = min(rows, key=lambda x: x["aggregate"]["fid"])
    bt = max(rows, key=lambda x: x["aggregate"]["top3"])
    print(f"== {lab}  ({len(rows)} evals)")
    for tag, x in [("best_fid", bf), ("best_top3", bt), ("last", rows[-1])]:
        a = x["aggregate"]
        print("   {:9s} ep{:>4}  FID {:.4f}  R@1 {:.4f}  R@2 {:.4f}  R@3 {:.4f}  MMD {:.4f}  Div {:.3f}".format(
            tag, x["epoch"], a["fid"], a["top1"], a["top2"], a["top3"], a["matching_score"], a["diversity"]))
    g = f"checkpoints/kit/{run}/model/gate"
    if os.path.isdir(g):
        print("   gate:", " ".join(sorted(os.listdir(g))))
