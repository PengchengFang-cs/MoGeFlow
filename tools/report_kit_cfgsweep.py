"""KIT x0-L1 1000ep cfg sweep: full curves for the three saved candidate checkpoints."""
import glob, json, os, re

S = "eval_results/kit_x0_cfgsweep_20260819"
CKPTS = [("best_top3", "ep900 best_top3（均衡点）"),
         ("best_fid", "ep790 best_fid"),
         ("gate/gate_ep0960_fid0.1332_top30.8338", "ep960 gate（R@1 最高）")]
for d, label in CKPTS:
    files = glob.glob(f"{S}/{d}/*.json")
    if not files:
        print(f"\n### {label}: 无数据"); continue
    rows = []
    for f in files:
        cfg = float(re.search(r"_cfg([0-9.]+)_", os.path.basename(f)).group(1))
        a = json.load(open(f))["aggregate"]
        a = {k: (v["mean"] if isinstance(v, dict) else v) for k, v in a.items()}
        rows.append((cfg, a))
    rows.sort()
    print(f"\n### {label}")
    print("| cfg | FID | R@1 | R@2 | R@3 | MMDist | Div |")
    print("|---|---|---|---|---|---|---|")
    for cfg, a in rows:
        print("| {:g} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} |".format(
            cfg, a["fid"], a["top1"], a["top2"], a["top3"], a["matching_score"], a["diversity"]))
