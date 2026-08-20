"""KIT x0 1000ep: full eval curve from ep500, marking which epochs have saved weights."""
import json, os, re
run = "kit_x0L1_ep1000_20260818"
rows = [json.loads(l) for l in open(f"checkpoints/kit/{run}/logs/full_eval.jsonl")]
g = f"checkpoints/kit/{run}/model/gate"
gate_eps = {int(re.search(r"ep(\d+)", f).group(1)) for f in os.listdir(g)} if os.path.isdir(g) else set()
bf = min(rows, key=lambda x: x["aggregate"]["fid"])["epoch"]
bt = max(rows, key=lambda x: x["aggregate"]["top3"])["epoch"]
print("| ep | FID | R@1 | R@2 | R@3 | MMDist | Div | 权重 |")
print("|---|---|---|---|---|---|---|---|")
for x in rows:
    if x["epoch"] < 500: continue
    a = x["aggregate"]; tags = []
    if x["epoch"] == bf: tags.append("best_fid.pt")
    if x["epoch"] == bt: tags.append("best_top3.pt")
    if x["epoch"] in gate_eps: tags.append("gate")
    if x["epoch"] == rows[-1]["epoch"]: tags.append("latest.pt")
    print("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {} |".format(
        x["epoch"], a["fid"], a["top1"], a["top2"], a["top3"], a["matching_score"], a["diversity"],
        "+".join(tags) if tags else "—"))
