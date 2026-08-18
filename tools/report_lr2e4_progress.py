"""One-shot progress table: lr2e-4@1000ep arms vs their 1e-4@600ep counterparts."""
import json

PAIRS = [
    ("hml3d_x0L1_ep1000_lr2e4_20260814", "hml3d_x0L1_e1_nopool_20260810", "L1"),
    ("hml3d_x0L7_ep1000_lr2e4_20260814", "hml3d_x0L7_normavg_nopool_20260810", "L7"),
    ("hml3d_x0L2_ep1000_lr2e4_20260814", "hml3d_x0L2_e1_pooltoken_20260810", "L2"),
]


def rows(run):
    try:
        return [json.loads(l) for l in open(f"checkpoints/t2m/{run}/logs/full_eval.jsonl")]
    except FileNotFoundError:
        return []


for new, old, lab in PAIRS:
    nr, orr = rows(new), rows(old)
    print(f"== {lab} lr2e-4@1000ep")
    if not nr:
        print("   还没到首评(ep100)")
        continue
    for x in nr:
        a = x["aggregate"]
        ref = next((y for y in orr if y["epoch"] == x["epoch"]), None)
        rs = ""
        if ref:
            ra = ref["aggregate"]
            rs = f'   (1e-4同期: {ra["fid"]:.4f}/{ra["top3"]:.4f})'
        print(f'   ep{x["epoch"]}: FID {a["fid"]:.4f}  R@3 {a["top3"]:.4f}  R@1 {a["top1"]:.4f}{rs}')
