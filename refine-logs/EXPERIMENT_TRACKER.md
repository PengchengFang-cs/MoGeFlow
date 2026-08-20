# MoGeFlow 实验执行清单（EXPERIMENT_TRACKER，持续更新）

> ARIS 约定：这是**执行清单**（TODO → RUNNING → DONE），回答"什么在跑、什么待跑、卡在哪"。永久结果与复现命令在 `EXPERIMENT_LOG.md`；计划与 claim 在 `refine-logs/EXPERIMENT_PLAN.md`。
> **与旧记录的关系**：`EXPERIMENT_TRACKER_20260610_170154.md` 是 6 月 inpainting 支线的清单（已搁置）；本文件自 2026-08-18 起追踪 MoGeFlow 主线。
> **更新规则**：每次发车 / 结束 / 失败 / 续跑，在**同一轮**里更新本表（附实证：日志行、GPU 占用、run 目录）。不得事后凭记忆补。
> 最后更新：2026-08-20

## 0. 此刻在跑（RUNNING）

（无）08-19 14:01 起卡池全空闲：1332153 / 1371418（H100，4d+）、1398694（H200×2，1d04h）、1357437（A100，2d16h）

## 1. 待办（TODO，按优先级）

| ID | 任务 | 依赖 | 说明 |
|---|---|---|---|
| T1–T3 | ~~Table 1 重填 / KIT 行 / Method 改写~~ → **08-20 取消**：论文全文保持 v 时代数字（用户裁定，见 PLAN B7） | — | 当天曾填入 cfg10 前 20 的 HML3D 行，已 git checkout 复原 |
| T4 | （可选）采样步数消融 8/16/32/64/96（纯 eval） | 用户拍板 | 几小时 |
| T5 | ScaMo/MotionMillion 作者名单再核；AI use statement（用户写）；bib-audit 提交仍在本地未 push | — | 只在指令下 push |

## 2. 已完成（DONE，倒序）

| 日期 | ID | 任务 | 结论一句话 | 详情 |
|---|---|---|---|---|
| 08-19 | B4 | x0L1 best_fid 100-seed × cfg9/cfg10（40 块，200 次评测，0 错误） | cfg9 0.0700/0.8673；cfg10 0.0676/0.8671（均值±CI 见 LOG#B4） | LOG#B4 |
| 08-19 | B8b | KIT x0-L1 1000ep `kit_x0L1_ep1000_20260818`（H100，中途换卡续跑） | best_fid ep790 0.1115/0.8210；best_top3 ep900 0.1217/**0.8366**；gate 3 个 | LOG#B8 |
| 08-17 | B8a | KIT x0-L1 600ep `kit_x0L1_e1_nopool_20260816`（H100 4h24m） | 相变晚 ~200ep；bf ep600 0.1374/0.8111，bt ep530 0.1576/0.8224；FID 反超 v 时代、R@3 差 0.03 | LOG#B8 |
| 08-16 | B5b | lr 2e-4 @1000ep 三臂（L1/L7/L2） | 负：FID 全部不如 1e-4/600ep，平衡点 0 | LOG#B5 |
| 08-14 | B6 | 区间引导 11 窗口（x0L1 bf，s=10） | 负：砍头段崩、砍尾段无差 | LOG#B6 |
| 08-14 | B5a | JiT 对齐三臂 600ep | 负：全面落后、后段走坏 | LOG#B5 |
| 08-13 | B3 | x0 cfg 全扫 1.5–11 × 10 检查点（100 JSON） | 联合最优 cfg 9–10 | LOG#B3 |
| 08-12 | B2 | x0-L 五臂 600ep | L1 最稳（12 平衡点）、L7 FID 最低、L8 垫底 | LOG#B2 |
| 08-11 | — | JiT 子版本参数（`--t_eps`、launcher）提交 28d1c67 | 仅配置，未训 | git |
| 08-10 | B1 | x0 硬编码 + 速度空间损失 + t clamp（8b78fac/064a97d） | 数学与微过拟合验证通过 | LOG#B1 |
| 08-09 | — | 损失外科（terminal/codebook CE、clean 删除）62139ad | 单一损失永久化 | memory `flow-only-loss-permanent` |
| 08-09 | B0 | 几何诊断 E1 δ / E2 白化 ρ / E3 RVQ 对照 | ρ 0.815→0.838；RVQ 只基层强 | `docs/PAPER_REVISION_CAMPAIGN_20260808.md` |
| 08-08 | B0 | KIT ep350 repeat-20 进论文；论文 3 审稿人闭环 | KIT 行 0.483/0.724/0.837/0.173/2.507/10.860 | 同上 |
| 08-07 | B0 | HML3D + KIT nopool_gate 重训（**terminal CE 意外=1.0**） | 产出 gate_ep0300 / kit ep320,350（v 时代） | memory + docs |
| 08-02→07 | B0 | pooled-routing 六 run + cfg 扫 72/72 + repeat-20 1/8 | nopool vs pooltoken 一次干净 A/B；FID 单次抖动 ~0.01 | `docs/POOLED_ROUTING_CAMPAIGN_20260802.md` |
| 08-01→02 | B0 | 文本编码器六 run 消融 | 三编码器 Top3 不可分；LLM2Vec-Qwen3 FID 最低 | `docs/ENCODER_ABLATION_FINAL_20260802.md` |
| 07-30 | B0 | 6 月检查点 repeat-20 复核 | 诚实值 0.0577±.0027 / 0.8608±.0025；continuous > nearest | `docs/TEXT_CONDITIONING_EXPERIMENTS_20260731.md` §1.2 |
| 05→07 | B0 | tokenizer 选定、主线、RVQ/code1024/encoder-latent 探索 | 见 `checkpoints/t2m/` 目录名与 `run_logs/` | 早期无系统文档 |

## 3. 失败 / 事故簿（原因 → 处置 → 规则）

| 日期 | 事件 | 处置 | 规则 |
|---|---|---|---|
| 08-17 | `pkill -f "python eval_codeflow..."` 在同节点误杀 2 个 seed 块（cfg10 s102/s107） | 回收重排，损失 ~2.5 H100h | 同节点只按 PID 杀 |
| 08-16 | L2 lr2e4 在 ep979 末死锁（主线程 100% CPU、GPU 0%） | kill → latest.pt 续跑 | 存活判据 = GPU + 日志新鲜度 |
| 08-14 | 会话重启带走 tmux → 三条 lr2e4 训练随 srun 死亡（约 3h） | latest.pt 无损续跑 | 同上；监控用 mtime 告警 |
| 08-13 | 1332155 到期，jit2 停在 ep359 | 1332152 续跑 | 发车前记到期时间 |
| 08-11 | x0L6 DDP 在 ep99→100 被 NCCL 10 分钟看门狗杀（rank0 评测 1h） | `--ddp_timeout_minutes 180` 续跑 | DDP + 长评测必须加 |
| 08-09 | 汇报"已发车"实未执行（lattice-B） | 记忆规则 | 每次发车同轮出示日志/GPU/目录 |
| 08-07 | gate 重训脚本漏掉 terminal 权重=0 → 训在 1.0 | 损失代码整段删除 | 见 `flow-only-loss-permanent` |
| 07-31 | LLM2Vec-Llama3 特征被分隔符污染 → §5 结论撤回 | 检查点删除、重跑 | `docs/TEXT_CONDITIONING_EXPERIMENTS_20260731.md` §6 |
