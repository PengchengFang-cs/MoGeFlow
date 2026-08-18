"""Full table for the limited-interval guidance sweep (x0L1 best_fid, s=10)."""
import glob
import json
import re

S = "eval_results/x0_cfgwin_20260814/hml3d_x0L1_e1_nopool_20260810"
REF = "eval_results/x0_cfg_sweep_20260812/hml3d_x0L1_e1_nopool_20260810/best_fid_test_s96_cfg10_tied_logits_continuous.json"


def agg(path):
    a = json.load(open(path))["aggregate"]
    return {k: (v["mean"] if isinstance(v, dict) else v) for k, v in a.items()}


def line(tag, a):
    print(f"| {tag} | {a['fid']:.4f} | {a['top1']:.4f} | {a['top2']:.4f} | {a['top3']:.4f} "
          f"| {a['matching_score']:.4f} | {a['diversity']:.3f} |")


rows = []
for f in glob.glob(f"{S}/*.json"):
    m = re.search(r"_win([0-9.]+?)-([0-9.]+?)\.json$", f)
    lo, hi = float(m.group(1)), float(m.group(2))
    rows.append((lo, hi, agg(f)))
rows.sort()

print("| 窗口 [lo,hi] | FID | R@1 | R@2 | R@3 | MMDist | Div |")
print("|---|---|---|---|---|---|---|")
line("恒定 (0,1) 参照", agg(REF))
for lo, hi, a in rows:
    line(f"[{lo:g}, {hi:g}]", a)
