"""One-shot: record the 2026-08-20 decision (paper keeps v-era numbers) in PLAN/TRACKER/LOG."""
import re

PLAN = "refine-logs/EXPERIMENT_PLAN.md"
TRACKER = "refine-logs/EXPERIMENT_TRACKER.md"
LOG = "EXPERIMENT_LOG.md"

s = open(PLAN).read()
old = ("论文里的消融表（key_ablations、decode/target）、MotionMillion 行、几何诊断全部保留 v 时代数字，"
       "**只重填 Table 1 的 HML3D 与 KIT 两行**（x0L1 100-seed + KIT x0）")
new = ("**用户 08-20 最终裁定：论文全文保持 v 时代数字不动**（本来就自洽）；x0 时代的全部结果作为后续工作的"
       "方向依据，不进本篇。此前『只重填 Table 1 两行』的计划作废。")
assert old in s
s = s.replace(old, new, 1)

old_b7 = """### 实验块 7（B7）：论文数字定稿（待 B4）
- **内容**：Table 1 HML3D 行 = B4 的 100-seed 均值±CI（选 cfg9 或 cfg10 由用户定）；Method 章改写为 x0（用户：论文暂缓，改完再写）；abstract 一句、AI use statement（用户写）
- **优先级**：必须 ⏳"""
new_b7 = """### 实验块 7（B7）：论文数字定稿 —— **取消（08-20）**
- **裁定**：论文保持 v 时代数字与 v 时代 Method，全文自洽，不做 x0 重填。08-20 曾按 cfg10 前 20 seed 填过 Table 1 / 摘要 / §4.2 四处，随即按用户指示 `git checkout` 全部复原，Overleaf 工作区干净（HEAD eb89474）。
- **理由（用户原话）**：全文保持以前的，本来就是自洽，最近跑的只是给未来一个指引
- **仍待用户自己处理**：AI use statement；bib-audit 提交（本地 eb89474，未 push）"""
assert old_b7 in s
s = s.replace(old_b7, new_b7, 1)

s = s.replace("- **论文 KIT 行**目前是 v 时代 ep350 r20 数字 → 待 B8 x0 结果替换（用户：只重填 Table 1 两行）",
              "- ~~论文 KIT 行待替换~~ → 08-20 取消：论文全文保持 v 时代数字")
s = s.replace("## 决策登记（已定案，不再重提）\n",
              "## 决策登记（已定案，不再重提）\n- **本篇论文使用 v 时代数字，全文不改**（08-20）；x0 时代结果 = 后续方向依据\n")
s = s.replace("> 最后更新：2026-08-18 06:40（KIT 1000ep 发车、seed-100 进行中）",
              "> 最后更新：2026-08-20（论文定稿口径：保持 v 时代数字；x0 结果转为后续依据）")
open(PLAN, "w").write(s)

s = open(TRACKER).read()
s = re.sub(r"\| T1 \|.*?\n\| T2 \|.*?\n\| T3 \|.*?\n",
           "| T1–T3 | ~~Table 1 重填 / KIT 行 / Method 改写~~ → **08-20 取消**：论文全文保持 v 时代数字"
           "（用户裁定，见 PLAN B7） | — | 当天曾填入 cfg10 前 20 的 HML3D 行，已 git checkout 复原 |\n",
           s, flags=re.S)
s = s.replace("> 最后更新：2026-08-19 14:05", "> 最后更新：2026-08-20")
open(TRACKER, "w").write(s)

s = open(LOG).read()
s = s.replace("> 最后更新：2026-08-19 14:05", "> 最后更新：2026-08-20")
block = """<a id="decision-20260820"></a>
## 决策 · 2026-08-20：论文保持 v 时代数字

用户裁定：全文保持以前的，本来就是自洽，最近跑的只是给未来一个指引。

- 当天曾按 cfg10 前 20 seed（0.592/0.783/0.867/0.064/2.582/9.702）填入 Table 1 HumanML3D 行、摘要、§4.2 两句，随即 `git checkout -- iclr2027_conference.tex` 全部复原；Overleaf 工作区干净（HEAD eb89474）。
- 触发点：`0.874/0.048` 同时出现在 tab:key_ablations 的 MoGeFlow-L 行与 tab:decode_target_ablations 的 continuous 行。只改主表 → 与『一个模型一个数字』冲突；一起改 → decode 消融结论被反转（nearest 0.058 vs x0 continuous 0.064），因为 0.058 是 v 时代同检查点的对照。保持全文 v 时代是唯一自洽解。
- **x0 时代的全部结果（B1–B8）转为后续工作的方向依据**，不进本篇。

---

<a id="b0"></a>"""
s = s.replace('<a id="b0"></a>', block, 1)
open(LOG, "w").write(s)
print("records updated")
