"""Full curves for the x0-era CFG sweep (cfg 1.5-11, five arms x best_fid/best_top3).

cfg 6 rows come from each run's training-time full_eval.jsonl (same protocol:
seed 42 / 96 steps / repeat 1 / test); every other cfg from
eval_results/x0_cfg_sweep_20260812/<run>/<kind>_test_s96_cfg<cfg>_*.json.
Run on a compute node:  srun --jobid=<J> --overlap --ntasks=1 bash -lc 'python tools/report_x0_cfg_sweep.py'
"""
import json
import os

RUNS = [
    ("hml3d_x0L1_e1_nopool_20260810", "x0L1 LLM2Vec nopool"),
    ("hml3d_x0L2_e1_pooltoken_20260810", "x0L2 LLM2Vec pooltoken"),
    ("hml3d_x0L7_normavg_nopool_20260810", "x0L7 raw-qwen3 nopool"),
    ("hml3d_x0L6_normavg_tok_r2_20260810", "x0L6 raw-qwen3 tok r2"),
    ("hml3d_x0L8_clipB_dual_20260810", "x0L8 CLIP-B dual"),
]
CFGS = [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
S = "eval_results/x0_cfg_sweep_20260812"


def fmt(a):
    return (f"| {a['fid']:.4f} | {a['top1']:.4f} | {a['top2']:.4f} | {a['top3']:.4f} "
            f"| {a['matching_score']:.4f} | {a['diversity']:.3f} |")


for run, label in RUNS:
    rows = [json.loads(l) for l in open(f"checkpoints/t2m/{run}/logs/full_eval.jsonl")]
    bf6 = min(rows, key=lambda r: r["aggregate"]["fid"])
    bt6 = max(rows, key=lambda r: r["aggregate"]["top3"])
    for kind, ref in [("best_fid", bf6), ("best_top3", bt6)]:
        print(f"\n### {label} — {kind} (ep{ref['epoch']})")
        print("| cfg | FID | R@1 | R@2 | R@3 | MMDist | Div |")
        print("|---|---|---|---|---|---|---|")
        for cfg in CFGS:
            if cfg == 6:
                print(f"| 6* {fmt(ref['aggregate'])[1:]}")
                continue
            tag = f"{cfg:g}"
            f = f"{S}/{run}/{kind}_test_s96_cfg{tag}_tied_logits_continuous.json"
            if not os.path.exists(f):
                print(f"| {tag} | MISSING |")
                continue
            a = json.load(open(f))["aggregate"]
            a = {k: (v["mean"] if isinstance(v, dict) else v) for k, v in a.items()}
            print(f"| {tag} {fmt(a)[1:]}")
print("\n(*cfg6 = training-time eval, same seed/steps; the row the checkpoint was selected by)")
