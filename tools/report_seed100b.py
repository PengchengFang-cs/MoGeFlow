"""Batch-2 (curated seeds) statistics for x0L1 best_fid @ cfg10, with per-group breakdown."""
import glob, json, math, statistics as st
S = "eval_results/x0L1_seed100b_20260819/cfg10"
KEYS = [("fid", "FID"), ("top1", "R@1"), ("top2", "R@2"), ("top3", "R@3"),
        ("matching_score", "MMDist"), ("diversity", "Div")]
recs = {}
for f in glob.glob(f"{S}/s*/*.json"):
    d = json.load(open(f))
    seeds = [int(x) for x in d["args"]["seed_list"].split(",") if x.strip()]
    for s, m in zip(seeds, d["repeat_metrics"]):
        recs[s] = {k: float(m[k]) for k, _ in KEYS}
if not recs:
    raise SystemExit("no data yet")
groups = [("常用", lambda s: s < 100 or s in {1234, 2020, 2021, 2022, 2023, 2024, 2025, 3407, 12345, 31415, 54321, 666, 777, 888, 999, 1000, 1024, 2048, 314, 555}),
          ("3位", lambda s: 100 <= s <= 999 and s not in {314, 555, 666, 777, 888, 999}),
          ("4位", lambda s: 1000 <= s <= 9999 and s not in {1000, 1024, 1234, 2048, 2020, 2021, 2022, 2023, 2024, 2025, 3407}),
          ("5位", lambda s: s >= 10000 and s not in {12345, 31415, 54321})]
def show(tag, seeds):
    n = len(seeds)
    if not n: return
    out = []
    for k, lab in KEYS:
        v = [recs[s][k] for s in seeds]
        mu = st.mean(v); sd = st.stdev(v) if n > 1 else 0.0
        out.append(f"{lab} {mu:.4f}±{1.96*sd/math.sqrt(n):.4f}")
    print(f"  {tag:6s} n={n:3d}: " + "  ".join(out))
print(f"batch2 已完成 {len(recs)} seeds")
show("全部", sorted(recs))
for tag, pred in groups:
    show(tag, sorted(s for s in recs if pred(s)))
