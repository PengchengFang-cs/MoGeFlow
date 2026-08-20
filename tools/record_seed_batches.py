"""Append the batch-2 (curated seeds) result to the living record."""

LOG = "EXPERIMENT_LOG.md"
TRACKER = "refine-logs/EXPERIMENT_TRACKER.md"

block = """
### B4b · 第二批 100 seed（精选 seed 列表，08-19 → 08-20 完成）

**动机**：第一批用连号 42–141，用户要求换一批"大家常用的 seed + 随机三/四/五位数"，验证结果不依赖 seed 的来源。
**seed 构成**（`tools/make_seedlist_batch2.py`，Random(20260819) 可复现，与第一批零重叠）：常用 25（0,1,2,3,7,1234,2020–2025,3407,12345,31415,54321,666,777,888,999,1000,1024,2048,314,555）+ 随机三位 25 + 四位 25 + 五位 25。
**实现**：为 eval CLI 新增 `--seed_list`（默认不填时与 `--seed`+`--repeat_times` 逐位一致，已冒烟验证）；5 seed/块 × 20 块；worker `run_logs/launch/x0L1_seed100b_worker_20260819.sh`；产物 `eval_results/x0L1_seed100b_20260819/cfg10/`。
**协议**：x0L1 best_fid ep600、cfg 10、96 步、test、bz 32。40 卡时，0 错误。

| 分组 | n | FID | R@1 | R@2 | R@3 | MMDist | Div |
|---|---|---|---|---|---|---|---|
| **全部** | 100 | **0.0680 ±0.0013** | 0.5922 ±0.0012 | 0.7831 ±0.0011 | **0.8670 ±0.0008** | 2.5809 ±0.0034 | 9.659 ±0.034 |
| 常用 seed | 25 | 0.0685 ±0.0025 | 0.5930 | 0.7846 | 0.8684 ±0.0014 | 2.5776 | 9.707 |
| 三位数 | 25 | 0.0674 ±0.0019 | 0.5923 | 0.7824 | 0.8667 ±0.0016 | 2.5841 | 9.618 |
| 四位数 | 25 | 0.0678 ±0.0031 | 0.5904 | 0.7835 | 0.8679 ±0.0018 | 2.5826 | 9.630 |
| 五位数 | 25 | 0.0682 ±0.0029 | 0.5932 | 0.7818 | 0.8652 ±0.0017 | 2.5795 | 9.681 |

**判读**：四个 seed 分组之间无系统性差异（FID 0.0674–0.0685，R@3 0.8652–0.8684，组间差远小于各自 CI）——**seed 的来源与位数不影响结果**。与第一批（连号 42–141，100 seed）对比：FID 0.0680 vs 0.0676、R@3 0.8670 vs 0.8671。两批共 200 个互不重叠的 seed 给出同一个答案，这个操作点的数字非常稳。
**复现**：`python tools/report_seed100b.py`（分组统计）、`tools/report_seed100.py`（第一批）。

---
"""

s = open(LOG).read()
anchor = '<a id="b8"></a>'
assert anchor in s and "B4b" not in s
s = s.replace(anchor, block.lstrip("\n") + "\n" + anchor, 1)
s = s.replace("> 最后更新：2026-08-20", "> 最后更新：2026-08-20（B4b 第二批 seed 收齐）", 1)
open(LOG, "w").write(s)

t = open(TRACKER).read()
hdr = "| 日期 | ID | 任务 | 结论一句话 | 详情 |\n|---|---|---|---|---|\n"
row = ("| 08-20 | B4b | 第二批 100 seed（常用 + 随机三/四/五位数），cfg10 | 0.0680/0.8670；四组无系统性差异；"
       "与第一批（0.0676/0.8671）一致 | LOG#B4b |\n")
assert hdr in t and "B4b" not in t
t = t.replace(hdr, hdr + row, 1)
t = t.replace("（无）08-19 14:01 起卡池全空闲",
              "（无）08-20 seed 第二批已收尾；卡池空闲", 1)
open(TRACKER, "w").write(t)
print("records updated")
