import json, glob, statistics as st, math
S="eval_results/x0L1_seed100_20260816"
for cfg in (9,10):
    recs={}
    for f in glob.glob(f"{S}/cfg{cfg}/s*/*.json"):
        d=json.load(open(f)); base=int(d["args"]["seed"])
        for i,m in enumerate(d["repeat_metrics"]):
            recs[base+i]=(float(m["fid"]),float(m["top3"]),float(m["top1"]),float(m["top2"]),float(m["matching_score"]),float(m["diversity"]))
    first=[s for s in range(42,62) if s in recs]
    allv=sorted(recs)
    def rep(seeds,tag):
        if not seeds: print(f"cfg{cfg} {tag}: 无数据"); return
        cols=list(zip(*[recs[s] for s in seeds])); names=["FID","R@3","R@1","R@2","MMDist","Div"]
        n=len(seeds)
        out=[]
        for nm,v in zip(names,cols):
            mu=st.mean(v); sd=st.stdev(v) if n>1 else 0
            out.append(f"{nm} {mu:.4f}±{1.96*sd/math.sqrt(n):.4f}")
        print(f"cfg{cfg} {tag} (n={n}, seed {seeds[0]}..{seeds[-1]}): "+"  ".join(out))
    rep(first,"前20(seed42-61)")
    rep(allv,"全部已完成")
    missing=[s for s in range(42,62) if s not in recs]
    if missing: print(f"   前20 缺 seed: {missing}")
