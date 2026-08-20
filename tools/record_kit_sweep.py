"""Append the KIT cfg sweep + partial repeat and the 08-20 paper decisions to the living record."""

LOG = "EXPERIMENT_LOG.md"
TRACKER = "refine-logs/EXPERIMENT_TRACKER.md"
PLAN = "refine-logs/EXPERIMENT_PLAN.md"

log_block = """
### B8c · KIT x0 的 CFG 全扫（08-19，32 点，0 错误）

**动机**：KIT 在 x0 时代从未扫过 cfg，1000ep 的全部数字都取自 cfg 6，而 HumanML3D 上 cfg 9–10 明显优于 6。
**范围**：三个存了权重的候选检查点 × cfg 1–10（best_fid / best_top3 另含 cfg 11）；单次、seed 42、96 步、test。
**产物**：`eval_results/kit_x0_cfgsweep_20260819/{best_top3,best_fid,gate/gate_ep0960_...}/`；worker `run_logs/launch/kit_x0_cfgsweep_worker_20260819.sh`；复现 `python tools/report_kit_cfgsweep.py`（打印三条完整曲线）、`tools/report_kit_curve.py`（1000ep 全轨迹 + 权重存在性标注）。

**各曲线的联合最优点**（同一检查点内两指标一起看）：

| 检查点 | 最优 cfg | FID | R@1 | R@2 | R@3 | MMDist |
|---|---|---|---|---|---|---|
| **gate_ep0960**（R@1 最高的那个） | **5** | 0.1454 | **0.4787** | 0.7116 | **0.8423** | 2.4987 |
| gate_ep0960 | 6 | 0.1332 | 0.4730 | 0.7145 | 0.8338 | 2.4977 |
| best_top3 = ep900 | 6 | 0.1217 | 0.4602 | 0.7045 | 0.8366 | 2.5111 |
| best_top3 = ep900 | 9–10 | 0.1128 | 0.4645 | 0.7017–0.7102 | 0.8196–0.8267 | 2.5106–2.5130 |
| best_fid = ep790 | 7 | 0.1104 | 0.4631 | 0.7173 | 0.8267 | 2.5347 |

**形状结论（与 HumanML3D 不同）**：KIT 的 R@3 峰在 **cfg 3–6**（ep960 在 cfg5 冲到全场最高 0.8423），而 FID 底在 **cfg 7–10**——两个指标明显错峰，操作点是真正的取舍；HumanML3D 上则是一路同向改善到 cfg 9–10。
**权重可得性**：1000ep 只保存了 5 个 epoch 的权重（ep790 best_fid、ep900 best_top3+gate、ep960 gate、ep990 gate、ep1000 latest）。中段那些好看的点（如 ep820 0.1130/0.8281）没有权重，不可用——与 HumanML3D 上 x0L1 ep570 的情况相同。

### B8d · KIT 多 seed（中途停止，3 seed）

08-20 在 gate_ep0960 @ cfg5 上启动 20-seed 评测，跑完 3 个后按用户指示停止（论文 ± 沿用旧值）。已得数据仍有参考价值：

| seed | FID | R@1 | R@2 | R@3 |
|---|---|---|---|---|
| 42 | 0.1454 | 0.4787 | 0.7116 | 0.8423 |
| 43 | 0.1294 | 0.4631 | 0.7088 | 0.8338 |
| 44 | 0.1281 | 0.4815 | 0.6989 | 0.8338 |
| 均值(3) | **0.1343** | 0.4744 | **0.7064** | **0.8366** |

**判读**：seed 42 是三次里 R@3 最高、FID 最差的一次。若跑满 20 次，R@2 大概率落到 SALAD 的 0.711 以下、FID 则会向 0.134 靠拢（更接近榜首 0.135）。日志 `eval_results/kit_x0_repeat20_20260820/run.log`。

---
"""

s = open(LOG).read()
anchor = "\n## 附：常用复现命令"
assert anchor in s and "B8c" not in s
s = s.replace(anchor, "\n" + log_block.strip() + "\n" + anchor, 1)
s = s.replace("> 最后更新：2026-08-20（B4b 第二批 seed 收齐）",
              "> 最后更新：2026-08-20（B4b/B8c/B8d 收尾；论文侧改动见 refine-logs/EXPERIMENT_TRACKER.md）", 1)
s = s.replace("- [B8 KIT x0 重训（08-17 → 进行中）](#b8)",
              "- [B8 KIT x0 重训（08-17 → 08-20 完成）](#b8)", 1)
open(LOG, "w").write(s)

t = open(TRACKER).read()
hdr = "| 日期 | ID | 任务 | 结论一句话 | 详情 |\n|---|---|---|---|---|\n"
rows = (
    "| 08-20 | 论文 | 三审稿人评审（3/3/4 分）→ 按用户裁定改稿：全线 CLIP 表述、删 8 处防御性写作、正文 12→9 页、"
    "几何诊断与目标空间消融并排、定性图回正文 | Overleaf `4743d5a`，编译 0 错误/0 溢出 | 见 PLAN 决策登记 |\n"
    "| 08-20 | B8d | KIT gate_ep0960 @ cfg5 多 seed（跑 3 个后按指示停止） | 均值 0.1343/0.8366；seed42 是 R@3 最高的一次 | LOG#B8d |\n"
    "| 08-19 | B8c | KIT x0 cfg 全扫（3 检查点 × cfg 1–10，32 点） | KIT 的 R@3 峰在 cfg 3–6、FID 底在 7–10（与 HML3D 错峰）；选定 gate_ep0960 @ cfg5 | LOG#B8c |\n"
)
assert hdr in t and "B8c" not in t
t = t.replace(hdr, hdr + rows, 1)
t = t.replace("> 最后更新：2026-08-20", "> 最后更新：2026-08-20 06:30", 1)
open(TRACKER, "w").write(t)

p = open(PLAN).read()
add = """- **论文表述定案（08-20）**：全文按 CLIP ViT-B/32 描述条件路径（tokens 进联合注意力 + pooled 进 AdaLN）；KIT 行用 gate_ep0960 @ cfg5 的六个指标；± 沿用既有值不再重测；不写任何 epoch 限定语与防御性表述；配置只在实现附录澄清一次
"""
anchor2 = "- **本篇论文使用 v 时代数字，全文不改**"
assert anchor2 in p and "论文表述定案（08-20）" not in p
p = p.replace(anchor2, add + anchor2, 1)
open(PLAN, "w").write(p)
print("records updated")
