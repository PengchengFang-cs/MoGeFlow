import json
RUNS=[("hml3d_x0L1_ep1000_lr2e4_20260814","hml3d_x0L1_e1_nopool_20260810","L1"),
      ("hml3d_x0L7_ep1000_lr2e4_20260814","hml3d_x0L7_normavg_nopool_20260810","L7"),
      ("hml3d_x0L2_ep1000_lr2e4_20260814","hml3d_x0L2_e1_pooltoken_20260810","L2")]
def rows(r): return [json.loads(l) for l in open(f"checkpoints/t2m/{r}/logs/full_eval.jsonl")]
def s(x):
    a=x["aggregate"]; return f"ep{x['epoch']:>4}  FID {a['fid']:.4f}  R@3 {a['top3']:.4f}  R@1 {a['top1']:.4f}"
for new,old,lab in RUNS:
    nr,orr=rows(new),rows(old)
    print(f"== {lab}  2e-4@1000ep ({len(nr)} evals, latest ep{nr[-1]['epoch']})   vs   1e-4@600ep")
    for tag,key,fn in [("best_fid","fid",min),("best_top3","top3",max)]:
        n=fn(nr,key=lambda x:x["aggregate"][key]); o=fn(orr,key=lambda x:x["aggregate"][key])
        print(f"   {tag:9s}: {s(n)}    |  1e-4: {s(o)}")
    print("   退火段轨迹 (2e-4):", "  ".join(f"ep{x['epoch']}:{x['aggregate']['fid']:.4f}/{x['aggregate']['top3']:.4f}" for x in nr if x['epoch']>=800 and x['epoch']%40==0))
    bal=[x for x in nr if x['aggregate']['fid']<0.10 and x['aggregate']['top3']>0.87]
    print(f"   平衡点(FID<0.10 & R@3>0.87): {len(bal)}",  " ".join(f"ep{x['epoch']}:{x['aggregate']['fid']:.4f}/{x['aggregate']['top3']:.4f}" for x in bal))
    obal=[x for x in orr if x['aggregate']['fid']<0.10 and x['aggregate']['top3']>0.87]
    print(f"   1e-4 同标准平衡点: {len(obal)}")
