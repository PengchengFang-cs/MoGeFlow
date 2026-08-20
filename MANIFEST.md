# Research Output Manifest — MoGeFlow

> ARIS 产物索引（`shared-references/output-manifest.md` 协议）。每生成/更新一个研究产物追加一行。时间为 BST。

| Timestamp | Skill / 来源 | File | Stage | Description |
|---|---|---|---|---|
| 2026-07-31 | 手工记录 | docs/TEXT_CONDITIONING_EXPERIMENTS_20260731.md | B0 | 文本条件实验记录（§5 已作废，§1.2 repeat-20 复核、§3/§6/§9 有效） |
| 2026-08-02 | 手工记录 | docs/ENCODER_ABLATION_FINAL_20260802.md | B0 | 文本编码器 6-run 消融定稿 |
| 2026-08-07 | 手工记录 | docs/POOLED_ROUTING_CAMPAIGN_20260802.md | B0 | pooled-routing 6-run + cfg 扫 + repeat-20 |
| 2026-08-09 | 手工记录 | docs/PAPER_REVISION_CAMPAIGN_20260808.md | B0/论文 | 审稿→修订闭环、几何诊断 E1–E3、决策登记 |
| 2026-08-13 | 本会话 | eval_results/x0_cfg_sweep_20260812/ (100 JSON) | B3 | x0 检查点 cfg 1.5–11 全扫 |
| 2026-08-14 | 本会话 | eval_results/x0_cfgwin_20260814/ (11 JSON) + tools/report_cfgwin_20260814.py | B6 | 区间引导窗口扫描 |
| 2026-08-16 | 本会话 | tools/report_lr2e4_{progress,best,final}.py | B5b | lr 2e-4@1000ep 汇总脚本 |
| 2026-08-17 | 本会话 | eval_results/kit_x0_diag_20260816/ | B8a | KIT ep150 cfg1 诊断（排除引导过强） |
| 2026-08-18 | 本会话 | eval_results/x0L1_seed100_20260816/ + tools/report_seed100.py | B4 | 100-seed 逐 seed 结果与统计（进行中） |
| 2026-08-18 06:40 | /research-pipeline（记录整理） | refine-logs/EXPERIMENT_PLAN.md | 全部 | claim→实验块→里程碑 的回溯计划 |
| 2026-08-18 06:40 | /research-pipeline（记录整理） | refine-logs/EXPERIMENT_TRACKER.md | 全部 | 执行清单：在跑/待办/已完成/事故簿 |
| 2026-08-18 06:40 | /research-pipeline（记录整理） | EXPERIMENT_LOG.md | 全部 | 永久记录：B0–B8 结果表、判读、复现命令 |
| 2026-08-18 06:40 | /research-pipeline（记录整理） | tools/report_x0_cfg_sweep.py | B3 | 复现全部 10 条 cfg 曲线的脚本 |
| 2026-05-17 | 早期 | PROJECT_DIRECTION_LOCK.md | 范围 | 硬边界：基础 T2M，无控制条件 |
| 2026-05-21→07-23 | 早期 | CODEFLOW_EXPERIMENT_LOG.md | B0 | 本 EXPERIMENT_LOG.md 的前身（PS-CF 定型、cfg/步数扫、continuous decode、KIT VQ） |
| 2026-05-24→07-12 | research-refine / pipeline | refine-logs/{round-*,FINAL_PROPOSAL,REFINEMENT_REPORT,PIPELINE_SUMMARY,EXPERIMENT_RESULTS}*.md | B0 | 早期 refine 运行与 07-12 pipeline 摘要；`_20260610_170154` 为已搁置的 inpainting 支线 |
| 2026-06-14 | research-review | CONTINUOUS_DECODE_RESEARCH_REVIEW_20260614.md | B0 | continuous decode 评审 |
| 2026-07-12 | pipeline Stage 4 | NARRATIVE_REPORT.md（+ 05-27/06-02/07-12 版本） | B0 | 叙事报告（pre-x0，待 T3 时刷新） |
| 2026-07-28 | 手工 | MEMORY.md（根目录） | 运维 | 07-28 发车与集群事故记录 |
| 2026-08-19 | 本会话 | eval_results/x0L1_seed100_20260816/（40 块 / 200 次评测） | B4 | HML3D 100-seed × cfg9/cfg10 完成 |
| 2026-08-19 | 本会话 | checkpoints/kit/kit_x0L1_ep1000_20260818/ + tools/report_kit_x0.py | B8b | KIT x0 1000ep 完成（gate 3 个） |
| 2026-08-20 | 本会话 | eval_results/x0L1_seed100b_20260819/（20 块 / 100 seed） + tools/{make_seedlist_batch2,report_seed100b}.py | B4b | 第二批精选 seed 统计（常用/3/4/5 位数各 25） |
