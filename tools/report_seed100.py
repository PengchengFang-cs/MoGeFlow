import json, glob, math, statistics as st
S="eval_results/x0L1_seed100_20260816"
KEYS=[("fid","FID"),("top1","R@1"),("top2","R@2"),("top3","R@3"),("matching_score","MMDist"),("diversity","Div")]
for cfg in (9,10):
    recs=[]
    for f in sorted(glob.glob(f"{S}/cfg{cfg}/s*/*.json")):
        d=json.load(open(f)); base=int(d["args"]["seed"])
        for i,m in enumerate(d["repeat_metrics"]):
            recs.append((base+i,{k:float(m[k]) for k,_ in KEYS}))
    recs.sort()
    n=len(recs)
    print(f"== cfg{cfg}: {n} seeds  (seed {recs[0][0]}..{recs[-1][0]})" if n else f"== cfg{cfg}: 0 seeds")
    if not n: continue
    print("| 指标 | mean | std | 95%CI(±1.96·std/√n) | min | max |")
    print("|---|---|---|---|---|---|")
    for k,lab in KEYS:
        v=[r[1][k] for r in recs]; mu=st.mean(v); sd=st.stdev(v) if n>1 else 0.0
        print(f"| {lab} | {mu:.4f} | {sd:.4f} | ±{1.96*sd/math.sqrt(n):.4f} | {min(v):.4f} | {max(v):.4f} |")
    s42=[r for r in recs if r[0]==42]
    if s42: print(f"   seed42 复现: FID {s42[0][1]['fid']:.4f} R@3 {s42[0][1]['top3']:.4f}")
