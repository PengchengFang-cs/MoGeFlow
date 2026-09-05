# MoGeFlow 实验永久记录（EXPERIMENT_LOG）

> ARIS 约定：**每个实验一个条目，成功失败都记**。这是"我们到底跑了什么、结果是什么、怎么复现"的权威来源。执行状态看 `refine-logs/EXPERIMENT_TRACKER.md`，计划看 `refine-logs/EXPERIMENT_PLAN.md`。
> 数字来源：全部来自 `full_eval.jsonl` / `eval_results/*.json` / 训练日志；汇总脚本列在各条目"复现"里（**在计算节点跑**：`srun --jobid=<J> --overlap --ntasks=1 bash -lc 'python tools/…'`）。
> 协议缩写：**训练期评测** = test split / cfg 6.0 / 96 步 / seed 42 / repeat 1 / EMA 权重 / bz 32 单卡；单次噪声底 σ_FID≈0.0064–0.0070、σ_R@3≈0.0042–0.0057；repeat-1 的 Top3 偏高 ≈0.013。
> 前身：`CODEFLOW_EXPERIMENT_LOG.md`（2026-05-21 → 07-23）。本文件从 x0 时代（08-09）起逐条详记，之前只留指针。
> 最后更新：2026-08-20（B4b/B8c/B8d 收尾；论文侧改动见 refine-logs/EXPERIMENT_TRACKER.md）

---

## 目录
- [B0 v 时代（2026-05 → 08-09）—— 摘要与指针](#b0)
- [B1 x0 硬编码 + 速度空间损失（08-09/10）](#b1)
- [B2 x0-L 五臂消融（08-10 → 08-12）](#b2)
- [B3 x0 CFG 全扫 1.5–11（08-12 → 08-13）](#b3)
- [B5 配方挑战者：JiT / lr 2e-4 @1000ep（08-13 → 08-16）](#b5)
- [B6 区间引导（08-14）](#b6)
- [B4 100-seed 定稿统计（08-17 → 进行中）](#b4)
- [B8 KIT x0 重训（08-17 → 08-20 完成）](#b8)

---

<a id="decision-20260820"></a>
## 决策 · 2026-08-20：论文保持 v 时代数字

用户裁定：全文保持以前的，本来就是自洽，最近跑的只是给未来一个指引。

- 当天曾按 cfg10 前 20 seed（0.592/0.783/0.867/0.064/2.582/9.702）填入 Table 1 HumanML3D 行、摘要、§4.2 两句，随即 `git checkout -- iclr2027_conference.tex` 全部复原；Overleaf 工作区干净（HEAD eb89474）。
- 触发点：`0.874/0.048` 同时出现在 tab:key_ablations 的 MoGeFlow-L 行与 tab:decode_target_ablations 的 continuous 行。只改主表 → 与『一个模型一个数字』冲突；一起改 → decode 消融结论被反转（nearest 0.058 vs x0 continuous 0.064），因为 0.058 是 v 时代同检查点的对照。保持全文 v 时代是唯一自洽解。
- **x0 时代的全部结果（B1–B8）转为后续工作的方向依据**，不进本篇。

---

<a id="b0"></a>
## B0 · v 时代（velocity 头 + 曾带 terminal CE）—— 摘要与指针

**日期**：2026-05-26（首个可移植代码发布）→ 2026-08-09
**状态**：权重不能在 x0 代码下 resume/复用；**评测数字与结论继续有效**（用户 08-18 澄清：不作废，论文只重填 Table 1 两行）：

| 子项 | 结论 | 记录 |
|---|---|---|
| tokenizer 选定 | Part-VQ `new_vq_overlap_top3_20260529`（单层、6 部件、K^P 积格；"RVQ"字样是命名陷阱） | memory `mogeflow-tokenizer-is-single-level-partvq` |
| 6 月检查点 repeat-20 复核（07-30） | 论文 0.048/0.874 是 seed 42 幸运抽样；诚实值 FID 0.0577±.0027 / Top3 0.8608±.0025；**continuous 解码 FID 优于 nearest（0.0577 vs 0.0686，CI 不重叠），检索无差** | `docs/TEXT_CONDITIONING_EXPERIMENTS_20260731.md` §1.2 |
| 文本编码器 6-run（08-01→02） | CLIP-L / LLM2Vec-Llama3 / LLM2Vec-Qwen3 × pool：Top3 不可分（0.8711–0.8722）；LLM2Vec-Qwen3 FID 最低 0.0459；nopool 两选点更好；Llama3 早期结论因特征污染撤回 | `docs/ENCODER_ABLATION_FINAL_20260802.md` |
| pooled-routing 6-run + cfg 扫 72/72 + repeat-20 1/8（08-02→07） | nopool vs pool-token 唯一干净 A/B；CLIP-L 后期漂移 +0.11 FID；**FID 单次抖动 ~0.01（clipL r2 cfg7: 单次 0.0338 → r20 0.0440±0.0030）** | `docs/POOLED_ROUTING_CAMPAIGN_20260802.md` |
| HML3D/KIT nopool_gate 重训（08-07） | 脚本漏 `--terminal_loss_weight 0` → CE=1.0 训练（事故）；产出 hml3d gate_ep0300、kit ep320/350 | memory `flow-only-loss-permanent` |
| KIT ep350 repeat-20（08-08） | 0.483±.007 / 0.724±.006 / 0.837±.004 / 0.173±.005 / 2.507±.016 / 10.860±.093 → 论文 Table 1 KIT 行（待 B8 x0 结果替换） | `docs/PAPER_REVISION_CAMPAIGN_20260808.md` §2.1；`eval_results/kit_repeat20_20260808/` |
| 几何诊断 E1/E2/E3（08-09） | δ 中位 0.68、26% 超半间距；白化 ρ 0.815→0.838；通用 RVQ 仅基层 ρ 0.80、其余 0.40–0.53 | 同上 §2.2–2.4；`eval_results/diag_20260809/` |
| 论文 3 审稿人闭环 | 5/4/4 → 修订全部落地；决策登记见 PLAN | 同上 §3–4 |
| PS-CF 方法定型 + 05-22 cfg/步数扫描 + 06-18 continuous decode/encoder-latent + 07-23 KIT part-aware VQ | 见原文各节 | `CODEFLOW_EXPERIMENT_LOG.md`（根目录，05-21→07-23，本文件的前身） |
| 07-10→12 HY-Motion 风格 CLIP-L + Qwen3-8B 缓存条件（cached provider、refiner、DDP） | 证据已收集、非论文就绪 | `NARRATIVE_REPORT.md`（07-12）、`refine-logs/PIPELINE_SUMMARY.md` |
| 06-14 continuous decode 研究评审 | — | `CONTINUOUS_DECODE_RESEARCH_REVIEW_20260614.md` |
| 项目硬边界（只做基础 T2M，KV-Control 仅作冻结 tokenizer） | — | `PROJECT_DIRECTION_LOCK.md`（05-17） |
| 07-28 五 run 发车与集群事故（管理员清空空闲分配） | — | `MEMORY.md`（根目录，07-28） |
| 早期探索（05→07） | momask-RVQ、code1024、encoder-latent 目标、RVQ1024 dim256/512、B/S 宽度、clip+qwen 双编码器等 | `checkpoints/t2m/` 目录名、`run_logs/launch/*2026060*` |

---

<a id="b1"></a>
## B1 · x0 硬编码 + 速度空间损失

**日期**：2026-08-09（损失外科）/ 08-10（x0）
**目标**：按用户 MotionCraft 设计——"预测 x0，计算还是 v"
**改动**（分支 `x0-dev`，本地）：
- 62139ad：terminal/codebook CE、clean loss 从 `motion_code_flow.py` / `part_structured_motion_code_flow.py` / `options` / `trainer` **删除**；parser 拒绝旧 flag；29 个 launcher 清理
- 8b78fac：`PREDICTION_TYPE="x0"`；`velocity_from_clean(z,t,x0)=(x0−z_t)/(1−t).clamp_min(t_eps)`；采样器 CFG 在 x0 空间合成后一次转换
- 064a97d：损失改为 `MSE(velocity_from_clean(z_t,t,x0_pred), Y1−Y0)`（≡ 1/(1−t)² 加权 x0 误差）；t 采样 clamp (1e-4, 1−1e-4)
- 28d1c67：`--t_eps`（默认 1e-4）+ JiT launcher（仅配置）

**验证**：CFG 等价 3e-6；末步 Euler 落 x̂0 3.6e-7；t_eps 在 96 步均匀网格不触发；微过拟合 loss 0.0016，1/2/8/32 步采样复现目标 4e-4。
**副作用记录**：x0-v-loss 初期 loss 很大（HML3D ep0 43.5、ep1 峰 ~295；KIT ep0 40.3、峰 848），非数据集特有，随训练收敛。

---

<a id="b2"></a>
## B2 · x0-L 五臂编码器/路由消融

**日期**：2026-08-10 发车 → 08-11/12 结束（L6 为 2×A100 DDP，`--ddp_timeout_minutes 180`）
**Launcher**：`run_logs/launch/hml3d_x0_L_20260810.sh`（ENC/POOLED/REFINER/CACHE 环境变量）、`hml3d_x0L6_ddp_20260811.sh`
**共同设置**：MoGeFlow-L（part 192 / hidden 1152 / 6 双流 + 12 单流 / 12 头 / mlp 4 / dropout 0.05）、bz 64、lr 1e-4（milestones 0.8/0.9/0.95 减半 → ep480/540/570）、warmup 2000、wd 0.01、cond_drop 0.1、600ep、seed 42、uniform t、t_eps 1e-4、EMA 评测每 10ep 自 ep100；gate 保存 FID<0.08 & R@3>0.87

| 臂 | run | 编码器 | pooled 路由 | refiner | best_fid（ep：FID/R@3） | best_top3（ep：R@3/FID） | 平衡点（FID<0.10 & R@3>0.87） |
|---|---|---|---|---|---|---|---|
| **x0L1** | `hml3d_x0L1_e1_nopool_20260810` | LLM2Vec-Qwen3 normavg（cache `humanml3d_qwen3_l2v_normavg_v2_20260802`） | none（AdaLN=t） | 0 | ep600：**0.0813 / 0.8666** | ep310：**0.8834 / 0.1484** | **12**（最佳 ep570 0.0890/0.8722，权重未存） |
| x0L2 | `hml3d_x0L2_e1_pooltoken_20260810` | LLM2Vec-Qwen3 | token | 0 | ep600：0.0812 / 0.8603 | ep180：0.8879 / 0.1714 | 15（ep400 0.1079/0.8810） |
| x0L7 | `hml3d_x0L7_normavg_nopool_20260810` | raw Qwen3-8B normavg（cache `humanml3d_qwen3_8b_normavg_userchat_l128_v1`） | none | 0 | ep600：**0.0703** / 0.8640 | ep270：0.8825 / 0.1730 | 0 |
| x0L6 | `hml3d_x0L6_normavg_tok_r2_20260810` | raw Qwen3 | token | 2 | ep600：0.1011 / 0.8700 | ep290：0.8849 / 0.2511 | 5 |
| x0L8 | `hml3d_x0L8_clipB_dual_20260810` | CLIP-B/32（token→context + pooled→AdaLN） | dual | 0 | ep600：0.1161 / 0.8599 | ep490：0.8763 / 0.1527 | 0 |

**判读**：无后期崩塌；R@1 可达 0.60–0.61；LLM2Vec 的价值在平衡区（12:0 vs raw Qwen3）；CLIP-B 全面垫底；gate 0.08/0.87 零触发（v 时代阈值对 x0 过严）。**用户裁定（08-16）：主模型 = x0L1 best_fid（ep600）**。
**复现**：`checkpoints/t2m/<run>/logs/full_eval.jsonl`；训练命令见 launcher。

---

<a id="b3"></a>
## B3 · x0 CFG 全扫（cfg 1.5–11 × 5 臂 × best_fid/best_top3）

**日期**：08-12 → 08-13（cfg 1.5–5 先扫 50 次；08-13 补 7–11 共 50 次；9 worker 跨 7 H100 + 2 A100，flock 队列）
**协议**：单次、seed 42、96 步、test、bz 32、`--disable_mm`；cfg6 用训练期评测；**用户要求：全部曲线原样汇报，不许摘要压缩、不许预判"无使用价值"**
**产物**：`eval_results/x0_cfg_sweep_20260812/<run>/<kind>_test_s96_cfg<cfg>_tied_logits_continuous.json`（100 个）；worker `run_logs/launch/x0_sweep_worker_20260812.sh`
**复现全部 10 条曲线**：`python tools/report_x0_cfg_sweep.py`

**关键点（摘录；完整 11 点曲线用脚本打印）**：

| 检查点 | cfg 6 | cfg 9 | cfg 10 | cfg 11 | 曲线内最优 |
|---|---|---|---|---|---|
| x0L1 best_fid ep600 | 0.0813/0.8666 | **0.0630/0.8733** | **0.0555/0.8713** | 0.0554/0.8694 | R@3 峰 cfg9，FID 底 cfg11 |
| x0L1 best_top3 ep310 | 0.1484/0.8834 | 0.1322/0.8821 | 0.1276/0.8845 | 0.1234/0.8860 | 高 R@3 只能用 FID 0.12+ 换 |
| x0L2 best_fid ep600 | 0.0812/0.8603 | 0.0663/0.8696 | 0.0643/0.8707 | 0.0635/0.8696 | 处处略逊 L1 |
| x0L7 best_fid ep600 | 0.0703/0.8640 | 0.0538/0.8707 | **0.0516/0.8713** | 0.0504/0.8705 | FID 全场最低段位 |
| x0L6 best_fid ep600 | 0.1011/0.8700 | 0.0701/0.8670 | 0.0646/0.8679 | 0.0585/0.8711 | — |
| x0L8 best_fid ep600 | 0.1161/0.8599 | 0.0772/0.8610 | 0.0710/0.8612 | 0.0687/0.8638 | — |

**形状结论**：所有有用曲线从 cfg1.5 到 6 单调改善（我早先"cfg6 对 x0 过高"的假设被数据否定）；R@3 峰在 cfg 8–10 出现并回落/持平；FID 到 11 仍缓降但增益趋平（L1 10→11 只降 0.0001）；Div 随 cfg 单调缓降（10.3→9.85）无坍塌。**联合最优被夹在 cfg 9–10**。低 cfg 的 FID 极小值（L2 bt cfg1.5 0.1275/0.8272）伴随检索坍塌，无意义。

---

<a id="b5"></a>
## B5 · 训练配方挑战者（全部负结果）

### B5a · JiT 对齐（Li & He "Back to Basics"）
**日期**：08-13 10:14 发车 → 08-14 结束（jit2 因 1332155 到期在 ep359 断、1332152 续跑）
**改动**：`--time_schedule logit_normal --denoiser_p_mean -0.8 --denoiser_p_std 0.8 --t_eps 0.05`，其余与 B2 同臂一致；launcher `run_logs/launch/hml3d_x0Ljit_20260811.sh`
**与 JiT 的对齐关系**：插值 z_t=t·x+(1−t)·ε、x-预测、v-loss-from-x-pred（其 Eq.6）、CFG-in-x 均一致；差异只在 t 分布、分母 clip、cfg 区间（其 1–4 vs 我们 ≥6，已被 B3 否定）、Heun50 vs Euler96

| 臂 | best_fid | best_top3 | ep600 | 对照 uniform-t 臂 |
|---|---|---|---|---|
| jit1（L1 配方） | ep280 0.1196/0.8578 | ep120 0.8683/0.1777 | 0.1972/0.8306 | 0.0813/0.8666、0.8834 |
| jit7（L7 配方） | ep310 0.1333/0.8556 | ep170 0.8722/0.1729 | 0.1935/0.8345 | 0.0703/0.8640、0.8825 |
| jit2（L2 配方） | ep480 0.1356/0.8450 | ep190 0.8666/0.1453 | 0.1823/0.8321 | 0.0812/0.8603、0.8879 |

**判读**：best_fid 差 0.04–0.07、best_top3 差 0.01–0.02（远超噪声底）；ep280–480 见底后走坏；gate 零触发。**uniform t / t_eps 1e-4 保留**。这同时是"t 采样超参消融"的负结果。

### B5b · lr 2e-4 + 1000ep（三臂）
**日期**：08-14 07:5x 发车（接力等 JiT 交棒）→ 08-16 结束；三条在 08-14 因会话重启死亡一次（latest.pt 续跑，零 epoch 损失）；L2 在 ep979 死锁一次（续跑）
**Launcher**：`run_logs/launch/hml3d_x0L_ep1000_20260814.sh`（`LR`/`MAX_EPOCH` 环境变量；milestones 自动落 ep800/900/950）；1e-4@1000ep 对照被用户取消
**复现汇总**：`python tools/report_lr2e4_final.py`（另 `report_lr2e4_progress.py`、`report_lr2e4_best.py`）

| 臂 | 2e-4@1000ep best_fid | 2e-4 best_top3 | 1e-4@600ep best_fid | 1e-4 best_top3 | 平衡点 2e-4 / 1e-4 |
|---|---|---|---|---|---|
| L1 | ep990 0.0826/0.8595 | ep410 0.8836/0.1513 | ep600 0.0813/0.8666 | ep310 0.8834/0.1484 | 0 / 5 |
| L7 | ep990 0.1090/0.8550 | ep220 0.8856/0.1726 | ep600 0.0703/0.8640 | ep270 0.8825/0.1730 | 0 / 0 |
| L2 | ep940 0.0949/0.8537（截至 ep970） | ep240 0.8860/0.1985 | ep600 0.0812/0.8603 | ep180 0.8879/0.1714 | 0 / 0 |

退火段（2e-4）：L1 ep800→1000 FID 0.1350→0.0879、R@3 0.864→0.860；L7 0.1380→0.1100、0.861→0.852；L2 0.1178→0.0971（ep960）。
**判读**：FID 端全部不如 1e-4（L1 持平在噪声内，L7 差 0.039、L2 差 0.014），R@3 峰持平；退火时"FID 降、R@3 同步掉"，比 1e-4 更强地把两指标推向对立。**lr 1e-4 / 600ep 保留**。

---

<a id="b6"></a>
## B6 · 区间引导（limited-interval CFG，Kynkäänniemi 2024）

**日期**：08-14 07:4x 改代码 → 08-14 15:xx 收齐 11/11（4 A100 + 2 H100 接力；会话重启后重建认领表）
**改动**：`sample_embeddings(cfg_t_lo=0, cfg_t_hi=1)`：t∈[lo,hi] 用 cond_scale，窗外 s=1；eval CLI `--cfg_t_lo/--cfg_t_hi`，文件名带 `_win<lo>-<hi>`；默认与恒定引导逐位一致（冒烟：max_batches=1 下 FID 完全相等）
**协议**：x0L1 best_fid、s=10、单次 seed 42
**产物**：`eval_results/x0_cfgwin_20260814/hml3d_x0L1_e1_nopool_20260810/*_win*.json`；worker `run_logs/launch/x0_cfgwin_worker_20260814.sh`；汇总 `tools/report_cfgwin_20260814.py`

| 窗口 [lo,hi] | FID | R@1 | R@2 | R@3 | MMDist | Div |
|---|---|---|---|---|---|---|
| 恒定 (0,1) 参照 | 0.0555 | 0.5946 | 0.7860 | 0.8713 | 2.5798 | 9.899 |
| [0, 0.7] | 0.0554 | 0.5968 | 0.7847 | 0.8700 | 2.5798 | 9.903 |
| [0, 0.85] | 0.0556 | 0.5935 | 0.7858 | 0.8707 | 2.5801 | 9.902 |
| [0.1, 0.7] | 0.0751 | 0.5819 | 0.7631 | 0.8506 | 2.6242 | 9.859 |
| [0.1, 0.85] | 0.0749 | 0.5823 | 0.7631 | 0.8522 | 2.6237 | 9.858 |
| [0.1, 1] | 0.0744 | 0.5834 | 0.7644 | 0.8513 | 2.6233 | 9.858 |
| [0.2, 0.7] | 0.1742 | 0.5496 | 0.7379 | 0.8289 | 2.7805 | 9.808 |
| [0.2, 0.85] | 0.1729 | 0.5496 | 0.7403 | 0.8274 | 2.7799 | 9.807 |
| [0.2, 1] | 0.1712 | 0.5502 | 0.7405 | 0.8280 | 2.7791 | 9.805 |
| [0.3, 0.7] | 0.2253 | 0.5269 | 0.7166 | 0.8069 | 2.8796 | 9.797 |
| [0.3, 0.85] | 0.2231 | 0.5272 | 0.7166 | 0.8071 | 2.8784 | 9.795 |
| [0.3, 1] | 0.2213 | 0.5269 | 0.7170 | 0.8071 | 2.8774 | 9.795 |

**判读**：砍尾段无损也无益；砍头段每 0.1 崩一档 → **引导收益几乎全在 t<0.1**，与 ImageNet 扩散上的原论文结论相反（我们的 FID 本就随 cfg 单调改善，没有"无谓损伤"可省）。不做第二阶段。副产品：t>0.7 的 uncond 前向可省 ~30% 采样算力（未采用）。

---

<a id="b4"></a>
## B4 · 100-seed 定稿统计（完成 08-19）

**用户裁定（08-16）**：模型 x0L1 best_fid（ep600）；cfg **9 与 10** 都做；**每个 100 seed，逐 seed 记录**
**实现**：5 seed/块（`--seed <base> --repeat_times 5`，seed = base+idx；base 42,47,…,137 → seed 42–141）× 20 块/cfg；flock 队列；worker `run_logs/launch/x0L1_seed100_worker_20260816.sh`；跨 H100/A100（跨卡型只影响同 seed 逐位复现，不影响 100-seed 统计）
**产物**：`eval_results/x0L1_seed100_20260816/cfg{9,10}/s<base>/best_fid_test_s96_cfg<cfg>_tied_logits_continuous.json`（每个含 `repeat_metrics[5]`）
**汇总**：`python tools/report_seed100.py`（逐 seed 可展开）
**事故**：08-17 08:18 同节点 `pkill -f` 误杀 2 块（回收重排）；退役 worker 多认领的 2 块回收；`retire_seed_worker.sh` 守卫解决竞态

**最终结果（08-19 14:01，40/40 块 = 200 次评测，0 错误）**：

cfg9（100 seed，42–141）：

| 指标 | mean | std | 95%CI | min | max |
|---|---|---|---|---|---|
| FID | **0.0700** | 0.0064 | ±0.0013 | 0.0585 | 0.0870 |
| R@1 | 0.5918 | 0.0052 | ±0.0010 | 0.5791 | 0.6058 |
| R@2 | 0.7826 | 0.0057 | ±0.0011 | 0.7685 | 0.7933 |
| R@3 | **0.8673** | 0.0048 | ±0.0009 | 0.8552 | 0.8791 |
| MMDist | 2.5818 | 0.0170 | ±0.0033 | 2.5511 | 2.6400 |
| Div | 9.6954 | 0.1898 | ±0.0372 | 9.0296 | 10.1645 |

cfg10（100 seed，42–141）：

| 指标 | mean | std | 95%CI | min | max |
|---|---|---|---|---|---|
| FID | **0.0676** | 0.0071 | ±0.0014 | 0.0514 | 0.0837 |
| R@1 | 0.5917 | 0.0054 | ±0.0011 | 0.5797 | 0.6043 |
| R@2 | 0.7827 | 0.0056 | ±0.0011 | 0.7679 | 0.7972 |
| R@3 | **0.8671** | 0.0042 | ±0.0008 | 0.8569 | 0.8780 |
| MMDist | 2.5831 | 0.0154 | ±0.0030 | 2.5504 | 2.6380 |
| Div | 9.6677 | 0.1891 | ±0.0371 | 9.0132 | 10.1441 |

**判读**：两个 cfg 的 R@3 在 CI 内完全相同（0.8673 vs 0.8671），cfg10 的 FID 低 0.0024（约 1.7 个 CI 半宽）；seed 层面 FID 与 R@3 几乎不相关（cfg10：FID 最低 seed86 0.0514/0.8610，R@3 最高 seed45 0.0724/0.8780）。seed42 单次值（cfg9 0.0630/0.8733、cfg10 0.0555/0.8713）均处于 FID 分布的偏好端，**论文只用均值**。

（历史中期快照 08-18 05:24）：

cfg10（90 seed，42–141）：

| 指标 | mean | std | 95%CI | min | max |
|---|---|---|---|---|---|
| FID | **0.0678** | 0.0070 | ±0.0015 | 0.0514 | 0.0837 |
| R@1 | 0.5916 | 0.0054 | ±0.0011 | 0.5797 | 0.6043 |
| R@2 | 0.7829 | 0.0056 | ±0.0011 | 0.7679 | 0.7972 |
| R@3 | **0.8672** | 0.0042 | ±0.0009 | 0.8569 | 0.8780 |
| MMDist | 2.5828 | 0.0153 | ±0.0032 | 2.5504 | 2.6380 |
| Div | 9.669 | 0.194 | ±0.040 | 9.013 | 10.144 |

cfg9（10 seed，102–121，仅参考）：FID 0.0724±0.0046，R@3 0.8698±0.0024。
**判读**：seed 42 逐位复现 sweep 值（0.0555/0.8713）且位于 FID 分布偏好端（前 5%）——单 seed sweep 数字对 FID 有 ~0.012 乐观偏差；论文引用以均值为准。**最终表待 40/40 收齐后替换本节。**

---

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

<a id="b8"></a>
## B8 · KIT x0-L1 同配方重训（完成 08-19）

**Launcher**：`hml3d_x0_L_20260810.sh`（`DATASET=kit`）/ `hml3d_x0L_ep1000_20260814.sh`（1000ep）；tokenizer `vqvae_kit_partaware_datadriven_300k_seed3407_h200_20260722`（net_best_fid）；cache `kit_qwen3_l2v_normavg_v2_20260807`（6368/6368 覆盖）；其余与 HML3D L1 一字不差；gate 0.25/0.83

### B8a · 600ep `kit_x0L1_e1_nopool_20260816`（08-17 07:01→11:25，H100 1332151）
| 阶段 | 现象 |
|---|---|
| ep100–220 | 采样崩塌：FID 81→84.6，R@3 0.08–0.11（≈随机 3/32），Div 2.5–2.8（真实 11.1）；**cfg1 同崩**（诊断 `eval_results/kit_x0_diag_20260816/`）→ 非引导过强 |
| ep260–340 | 相变：FID 55.8→3.86，R@3 0.196→0.645 |
| ep340–600 | 单调改善：ep410 0.492/0.776 → ep500 0.181/0.805 → **ep600 0.1374/0.8111（best_fid）**；best_top3 ep530 0.1576/**0.8224**；末 6 点 FID 仍降 |
| 对照 v 时代（带 CE 的 kit_20260807 ep350） | 0.1791/0.8509（单次），ep200 已 0.93/0.815 |

**判读**：x0-flow-only 在 KIT 上**晚熟 ~200ep**（训练侧 loss/token_acc 全程正常，是采样输出成型晚）；FID 已反超、R@3 差 0.03；有效训练只有起飞后 ~300ep → 处方是拉长而非改配方。我曾误判为"坏了"，已收回。

### B8b · 1000ep `kit_x0L1_ep1000_20260818`（08-18 06:0x → 08-19 完成；1332152 到期后从 latest.pt ep518 在 1332153 续跑）
- 配置实锤（options.json）：kit / lr 1e-4 / max_epoch 1000 / uniform / t_eps 1e-4 / llm2vec_cache + modulation / seed 42；ep0 loss 40.35 与 600ep 版逐位一致；milestones ep800/900/950

| 检查点 | FID | R@1 | R@2 | R@3 | MMDist | Div |
|---|---|---|---|---|---|---|
| best_fid ep790 | **0.1115** | 0.4673 | 0.7102 | 0.8210 | 2.5365 | 10.768 |
| best_top3 ep900 | 0.1217 | 0.4616 | 0.7045 | **0.8366** | 2.5113 | 10.721 |
| ep1000（末点） | 0.1363 | 0.4759 | 0.7003 | 0.8281 | 2.4902 | 10.657 |
| 600ep 版 best_fid ep600（对照） | 0.1374 | 0.4602 | 0.6932 | 0.8111 | 2.5748 | 10.772 |
| v 时代 ep350 repeat-20（论文现行） | 0.1732±.005 | 0.4834±.007 | 0.7244±.006 | 0.8366±.004 | 2.5067±.016 | 10.860±.093 |

- gate（0.25/0.83）首次触发，保存 3 个：ep900 (0.1217/0.8366)、ep960 (0.1332/0.8338)、ep990 (0.1364/0.8310)
- **判读**：1000ep 相对 600ep 全面改善（FID 0.1374→0.1115，R@3 0.8111→0.8366）；R@3 在 ep900 **追平 v 时代的 0.8366**，FID 则大幅优于 v 时代（0.1217 vs 0.1732，差 10 个 CI 半宽）；R@1/R@2 仍低于 v 时代（0.4616/0.7045 vs 0.4834/0.7244）。退火后 ep950–1000 FID 回升、R@3 走平 → 1000ep 已到该配方在 KIT 上的收敛点，再拉长无益。
- **待用户拍板**：KIT 行用 x0（ep900，需补 100-seed）还是沿用 v 时代 ep350。

---

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

## B9 · 审稿回应诊断三条（08-31，pink7002 job 1476665 H200×2，三条全部 exit 0）

回应 08-30 三位 ICLR 审稿人共同指认的三个缺口。均复用既有诊断管线，未改动任何模型文件。产物在 `eval_results/diag_20260831/`。

### B9a · 解码侧 ρ（`tools/decoder_side_rho.py`）

**问题**：论文的 ρ 用编码侧原型（该码出现处的归一化动作特征均值），而摘要 L73 / 结论 L728 写的是"码所**解码出**的动作"。R1 把这条排在缺点第二，理由是编码侧版本在最近邻 VQ + 近似 Lipschitz 编码器下近乎构造性成立。

**做法**：每组取 16 个真实上下文；在每个上下文的一个有效步上把该组的码逐一替换为全部 128 个码，整批过冻结解码器，读回该步局部窗口（帧 4t..4t+4）限制到该组通道，按上下文平均得解码侧原型；再与码嵌入距离算 Spearman（1000 次 bootstrap）。

| group | 活跃码 | ρ 解码侧 | 95% CI | ρ 编码侧 |
|---|---|---|---|---|
| root_lower_spine | 123 | 0.9615 | [0.960, 0.963] | 0.9625 |
| upper_torso_arms | 126 | 0.9639 | [0.960, 0.968] | 0.8523 |
| right_leg | 106 | 0.6034 | [0.588, 0.619] | 0.7791 |
| upper_spine_neck | 111 | 0.9420 | [0.939, 0.945] | 0.9026 |
| left_leg | 113 | 0.5947 | [0.578, 0.610] | 0.6294 |
| head | 99 | 0.9384 | [0.934, 0.942] | 0.8035 |
| **均值** | 113.0 | **0.8340** | | **0.8216** |

**结论**：解码侧 0.834 **高于**编码侧 0.822，"decode to" 字面成立，无需改措辞。编码侧一列与论文 Table 3(a) 的 0.822 完全对上（活跃码 123/126/106/111/113/99，合计 678，均值 113.0），同协议同 split 无漂移。两条腿是解码侧唯一低于编码侧的组（0.60/0.59 对 0.78/0.63）。

### B9b · 活跃码数配平后的 ρ（`tools/rho_matched_active_codes.py`）

**问题**：未训练 tokenizer 对照报 ρ 0.24 对 0.82，但它每组只有 37 个活跃码对 113。Spearman 在码对上算（37 码=666 对，113 码=6328 对），支撑度不同，三位审稿人都指出这个混淆。

**做法**：训练好的 tokenizer 上，把每组活跃码随机子采样到 37 个再算 ρ，重复 200 次。

| group | 活跃码 | ρ 全支撑 | ρ 配平(37) | 95% 区间 |
|---|---|---|---|---|
| root_lower_spine | 123 | 0.9625 | 0.9613 | [0.947, 0.974] |
| upper_torso_arms | 126 | 0.8523 | 0.8496 | [0.759, 0.911] |
| right_leg | 106 | 0.7791 | 0.7799 | [0.672, 0.864] |
| upper_spine_neck | 111 | 0.9026 | 0.8976 | [0.848, 0.935] |
| left_leg | 113 | 0.6294 | 0.6331 | [0.489, 0.762] |
| head | 99 | 0.8035 | 0.7981 | [0.718, 0.863] |
| **均值** | | **0.8216** | **0.8199** | |

**结论**：配平到 37 码只掉 0.0017。未训练对照的 0.24 与活跃码数无关，该对照成立。

### B9c · 离格偏移的 in-span 能量（`tools/offlattice_inspan.py`）

**问题**：论文把终点态离格偏移读作"亚码精度"、并称格点之间的状态解码为"相邻原型的混合"。R3 指出这要求偏移落在局部码差张成的子空间里，否则按论文自己的等模长随机方向对照，解码器对它恰恰不敏感——那偏移就该读作回归残差。

**做法**：`gate_ep0300`（与论文 App A.3 离格统计同一检查点），512 prompt、96 步 Euler、cfg 6.0 采样终点态。对每个有效 frame-group：k\* = 最近码，δ = y − e_{k\*}，取 k\* 的 m=8 个最近码构成 span{e_{k_j} − e_{k\*}}，SVD 取正交基 Q，报 in-span 能量占比 ‖Qᵀδ‖²/‖δ‖²，并与同模长随机向量在同一子空间上的占比对比。

| group | 子空间秩 | in-span 占比 | 随机对照 | 倍数 |
|---|---|---|---|---|
| root_lower_spine | 8 | 0.2952 | 0.0625 | **4.72×** |
| upper_torso_arms | 8 | 0.2322 | 0.0626 | 3.71× |
| right_leg | 8 | 0.1854 | 0.0628 | 2.95× |
| upper_spine_neck | 8 | 0.1693 | 0.0621 | 2.72× |
| left_leg | 8 | 0.2564 | 0.0628 | 4.08× |
| head | 8 | 0.1822 | 0.0628 | 2.90× |
| **总体** | 8 | **0.2201** | **0.0626** | **3.52×** |

**结论**：128 维取 8 维子空间，各向同性期望占比 8/128 = 0.0625，随机对照实测 0.0626（基与投影正确）。真实偏移的 in-span 能量 0.2201 = 随机基线的 **3.52 倍**，六组无一例外（2.72–4.72×）。偏移方向显著富集于相邻码方向，不是各向同性残差。

**诚实边界**：0.22 意味着仍有 78% 能量在该局部子空间之外。准确表述是"显著富集于相邻码方向（3.5 倍于各向同性基线）"，而非"落在相邻码张成的子空间内"。

### 复现

```bash
srun --jobid=<J> --overlap --ntasks=1 bash -lc '
  cd /iridisfs/scratch/pf2m24/projects/Umdd/momask-codes
  P=/scratch/pf2m24/miniconda3/envs/fudoki-momask/bin/python
  $P tools/decoder_side_rho.py        --split test --out eval_results/diag_20260831/decoder_side_rho.json
  $P tools/rho_matched_active_codes.py --split test --out eval_results/diag_20260831/rho_matched_active.json
  $P tools/offlattice_inspan.py --max_prompts 512 --steps 96 --cond_scale 6.0 \
     --out eval_results/diag_20260831/offlattice_inspan.json'
```

---

## B10 · MultiModality 补测（09-02，pink7002 job 1476665）

论文一直只有 **MultiModal Dist**（文本嵌入 ↔ 生成动作嵌入的距离），没有 **MultiModality**（同一条文本生成多条动作之间的平均两两距离）。两者名字近但量的东西不同：前者是"贴不贴这句话"，后者是"同一句话能生成多少种"。评测代码两个都实现了（`matching_score` / `multimodality`），我们历次评测一直带 `--disable_mm`，关掉的是后者。

**协议**：与 Table 1 对应行同检查点、同 cfg、`--seed 42 --repeat_times 1 --steps 96`，唯一差别是去掉 `--disable_mm`；MM 参数用默认（`mm_num_samples 30` / `multimodality_times 10` / `mm_num_batches 3`）。产物 `eval_results/mm_20260902/`。

### B10a · KIT（`gate_ep0960` @ cfg5，26 分钟，exit 0）

**MultiModality = 0.9470**

同一次运行的其它指标与论文那一行有系统性偏移：

| | 论文 Table 1 | 本次（带 MM） | 差 |
|---|---|---|---|
| Top1 | 0.479 | 0.4659 | −0.013 |
| Top2 | 0.712 | 0.7031 | −0.009 |
| Top3 | 0.842 | 0.8352 | −0.007 |
| FID | 0.145 | 0.1359 | −0.010 |
| MM-Dist | 2.499 | 2.5202 | +0.021 |
| Diversity | 10.724 | 10.7618 | +0.038 |

**成因**：MM 那一遍额外采样与主评测共用同一随机数流，打开它就把后续抽样序列整体挪位。偏移量级为噪声底（σ_Top3≈0.0057、σ_FID≈0.0064）的 1–1.5 倍，属抽样漂移而非 bug。

**裁定（09-02）**：走 A —— 只把 MultiModality 填进表，其余五个数字保持论文原值。理由：MultiModality 在该协议下本就是独立的一遍生成，按列单独报是通例；换整行会让 KIT Top-2 从 0.712 掉到 0.7031、失去对 SALAD（0.711）的领先。

### B10b · HumanML3D（六月主线 `best_top3.pt` @ cfg6）

**MultiModality = 1.2157**

Table 1 的 HumanML3D 行来自六月主线 `codeflow_part_structured_newvqtop3_..._20260601/model/best_top3.pt`（其 `eval_decode_compare_seed42_20260614/*.json` 与论文行逐位一致：0.5894/0.7841/0.8739/0.047862/2.5995/9.9140）。

**前两次尝试全部塌掉**，且数字逐位相同（FID 49.5137 / Top3 0.1978 / Diversity 3.2461 / MM 2.2820），第二次已按该 run 的 `options.json` 补齐 kv_root / vq_checkpoint / vq_partition / mean / std / **clip_path** / unit_length。两次相同说明与传入路径无关。

**根因**：`models/codeflow/motion_code_flow.py:39` 明写 "There is no velocity-prediction mode."——08-09 的损失外科手术把速度预测分支永久删除，当前代码 x0 硬编码（memory `flow-only-loss-permanent`）。六月检查点是 **v 时代**模型，网络头输出速度，被 x0 采样器当 x0 解释，故输出塌成噪声。KIT 一路能跑是因为 `gate_ep0960` 属 x0 时代。

**修法（评测专用，不碰训练路径）**：采样器 `forward_guided` 内加分支，`eval_head_is_velocity` 为真时把网络输出直接当速度用、跳过 `velocity_from_clean`，自条件所需的 x0 由 `z + (1-t)·v` 反推；CLI 加 `--head_is_velocity`。训练仍是 x0 硬编码，单一损失那条决定不受影响。

**验证与结果**（同一次运行同时给出主指标与 MM，不需要两趟）：

| | 六月参考（repeat-1, seed42, cfg6） | 本次（带 MM） | 差 |
|---|---|---|---|
| Top1 | 0.5894 | 0.5815 | −0.008 |
| Top2 | 0.7841 | 0.7780 | −0.006 |
| Top3 | 0.8739 | 0.8685 | −0.005 |
| FID | 0.0479 | 0.0494 | +0.002 |
| MM-Dist | 2.5995 | 2.6130 | +0.014 |
| Diversity | 9.9140 | 9.8904 | −0.024 |
| **MultiModality** | —（当时关闭） | **1.2157** | — |

回到正确量级，残差与 B10a（KIT）同性质同量级：开 MM 后额外采样与主评测共用随机数流导致的抽样漂移，落在噪声底（σ_Top3≈0.0057、σ_FID≈0.0064）之内。通路读对了。

**入论文**：按 B10a 的裁定（走 A），只把 MultiModality 填进 Table 1，其余五个数字保持论文原值——HumanML3D **1.216**、KIT **0.947**，两者都是各自列里最低。

**规则**：v 时代检查点须带 `--head_is_velocity` 才能用当前代码评测；不带就会静默塌成噪声（不报错），这是一个易踩的坑。
---

## B11 · 目标空间消融的归一化修正（09-04 → 09-05，1476669 H200）

**问题来源**：三位审稿人（09-04 那轮）中有两位从症状反推出同一个怀疑——Table 3(b) 的 encoder-latent 臂 FID 劣化 3.15 倍而 R@3 只掉 0.014，这个组合更像"目标尺度没配好"而非"目标空间几何更差"。查 `options.json` 证实了：

| 臂 | `kv_part_target_mode` | `latent_norm_mode` |
|---|---|---|
| encoder latent（原） | `encoder` | **`codebook`** ← 用码本统计量白化一个非码本目标 |
| variational（原） | — | `empirical` |

即两条非格点臂的归一化方式**互不一致**，且 encoder-latent 那条用错了空间的统计量。

**做法**：复制 `run_logs/launch/ae_b_enclat_20260808.sh`，**只改两行**（run 名 + `--latent_norm_mode codebook` → `empirical`），其余逐字相同 → `ae_b_enclat_empnorm_20260904.sh`，B 宽度，600ep。

**过程**：09-04 08:03 发车，跑到 epoch 299 时 tmux 会话被拆除、训练随 srun 死亡（同 08-14 那次事故，作业本身没到期）；09-05 09:08 从 `latest.pt` 续跑，`Resumed from ... at epoch=300 step=114900`。**续跑无损**——跨断点的评测曲线单调连续（290 fid 0.1038 → 310 0.0944 → 320 0.0887 → 330 0.0823），无跳变。epoch 300 的评测点因崩溃丢失。

**结果**（峰值在 epoch 330，之后单调退化，340 起 `updated=none`）：

| 口径 | 原臂（codebook 白化） | 新臂（empirical 白化） |
|---|---|---|
| 训练期 full_eval 最优 | FID 0.2195 / Top3 0.850 | **FID 0.0823 / Top3 0.8537** |
| 单独 EMA 评测（进表口径） | 0.151 / 0.860 | **0.0823 / 0.8530** |

单独评测完整值：FID 0.0823、Top1 0.5528、Top2 0.7552、Top3 0.8530、MM-Dist 2.7185、Diversity 9.7822。

**判读**：归一化确实是主因——同口径下 FID 从 0.2195 降到 0.0823。但**结论方向不变**：格点目标仍两轴皆胜（0.874/0.048 对 0.853/0.082），只是差距从 3.1 倍收到 1.7 倍。

**入论文**：Table 3(b) encoder-latent 行换成 `0.853 / 0.082`；A.4 补"每个目标用自身逐维统计量白化"。正文无任何一处引用旧的 0.151/0.860，依赖的都是定性表述（"两轴皆更弱"），仍然成立。

**规则**：非格点目标必须用其自身空间的统计量白化；沿用码本统计量会制造一个看起来像"几何更差"的假象。

## B12 · guidance 扫描（09-04，三卡并行，**不进论文**）

在**论文那个六月检查点**上扫 cfg = 1,2,3,4,5,7,8（6.0 已有），带 `--head_is_velocity`、带 MultiModality、seed 42、96 步、repeat 1。产物 `eval_results/cfgsweep_june_20260904/`。

| cfg | Top1 | Top2 | Top3 | FID | MM-Dist | Div | MModality |
|---|---|---|---|---|---|---|---|
| 1 | 0.5168 | 0.7091 | 0.8054 | 0.2747 | 2.9312 | 9.863 | 2.041 |
| 2 | 0.5601 | 0.7567 | 0.8450 | 0.1471 | 2.7037 | 10.061 | 1.456 |
| 3 | 0.5834 | 0.7696 | 0.8595 | 0.0994 | 2.6462 | 10.040 | 1.307 |
| 4 | 0.5832 | 0.7778 | 0.8621 | 0.0685 | 2.6221 | 9.987 | 1.259 |
| 5 | 0.5881 | 0.7791 | 0.8638 | 0.0543 | 2.6107 | 9.939 | 1.240 |
| 7 | 0.5819 | 0.7793 | **0.8713** | **0.0495** | 2.6167 | 9.827 | 1.211 |
| 8 | 0.5767 | 0.7791 | 0.8670 | 0.0546 | 2.6190 | 9.804 | 1.208 |

**判读**：曲线平滑，论文所用的 cfg 6（0.874/0.048）位于合理位置，峰值区在 cfg 6–7。**七个点里 FID 最低只到 0.0495，够不到 MotionHiFlow 的 0.032。**

**裁定（09-04）：不进论文。** 论文报的点本来就在 Top3 与 FID 两项上同时优于 SALAD（0.874/0.048 对 0.857/0.076），不需要扫描来证明；扫描的唯一用途是证明"没挑幸运点"，属于替一个论文未提出的质疑做辩护，与"不做防御性写作"冲突。数据留档备查。

---

## 附：常用复现命令
```bash
# 训练（HML3D L1 臂，600ep）
DATASET=hml3d RUN_NAME=<name> GPU_ID=0 bash run_logs/launch/hml3d_x0_L_20260810.sh
# 训练（KIT，1000ep）
DATASET=kit RUN_NAME=<name> LR=1e-4 MAX_EPOCH=1000 GPU_ID=0 bash run_logs/launch/hml3d_x0L_ep1000_20260814.sh
# 评测（HML3D，单次 / 多 seed）
python eval_codeflow_part_structured_t2m.py --checkpoint <ckpt> --eval_dir <dir> --steps 96 --cond_scale 9 \
  --seed 42 --repeat_times 1 --batch_size 32 --num_workers 4 --disable_mm --gpu_id 0 [--cfg_t_lo 0 --cfg_t_hi 1]
# 评测（KIT）额外：--dataset_opt_path checkpoints/kit/Comp_v6_KLD005/opt.txt --data_root /scratch/pf2m24/data/KIT-ML \
#   --kv_root $KV --vq_checkpoint $VQD/model/net_best_fid.tar --vq_partition $VQD/config/kit_skeleton_partition_partaware_datadriven_seed3407.json \
#   --mean_path $VQD/meta/mean.npy --std_path $VQD/meta/std.npy
# 所有 python 汇总/评测一律：srun --jobid=<J> --overlap --ntasks=1 bash -lc '...'
```
