# MoGeFlow 实验永久记录（EXPERIMENT_LOG）

> ARIS 约定：**每个实验一个条目，成功失败都记**。这是"我们到底跑了什么、结果是什么、怎么复现"的权威来源。执行状态看 `refine-logs/EXPERIMENT_TRACKER.md`，计划看 `refine-logs/EXPERIMENT_PLAN.md`。
> 数字来源：全部来自 `full_eval.jsonl` / `eval_results/*.json` / 训练日志；汇总脚本列在各条目"复现"里（**在计算节点跑**：`srun --jobid=<J> --overlap --ntasks=1 bash -lc 'python tools/…'`）。
> 协议缩写：**训练期评测** = test split / cfg 6.0 / 96 步 / seed 42 / repeat 1 / EMA 权重 / bz 32 单卡；单次噪声底 σ_FID≈0.0064–0.0070、σ_R@3≈0.0042–0.0057；repeat-1 的 Top3 偏高 ≈0.013。
> 前身：`CODEFLOW_EXPERIMENT_LOG.md`（2026-05-21 → 07-23）。本文件从 x0 时代（08-09）起逐条详记，之前只留指针。
> 最后更新：2026-08-18 06:40

---

## 目录
- [B0 v 时代（2026-05 → 08-09）—— 摘要与指针](#b0)
- [B1 x0 硬编码 + 速度空间损失（08-09/10）](#b1)
- [B2 x0-L 五臂消融（08-10 → 08-12）](#b2)
- [B3 x0 CFG 全扫 1.5–11（08-12 → 08-13）](#b3)
- [B5 配方挑战者：JiT / lr 2e-4 @1000ep（08-13 → 08-16）](#b5)
- [B6 区间引导（08-14）](#b6)
- [B4 100-seed 定稿统计（08-17 → 进行中）](#b4)
- [B8 KIT x0 重训（08-17 → 进行中）](#b8)

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
## B4 · 100-seed 定稿统计（进行中）

**用户裁定（08-16）**：模型 x0L1 best_fid（ep600）；cfg **9 与 10** 都做；**每个 100 seed，逐 seed 记录**
**实现**：5 seed/块（`--seed <base> --repeat_times 5`，seed = base+idx；base 42,47,…,137 → seed 42–141）× 20 块/cfg；flock 队列；worker `run_logs/launch/x0L1_seed100_worker_20260816.sh`；跨 H100/A100（跨卡型只影响同 seed 逐位复现，不影响 100-seed 统计）
**产物**：`eval_results/x0L1_seed100_20260816/cfg{9,10}/s<base>/best_fid_test_s96_cfg<cfg>_tied_logits_continuous.json`（每个含 `repeat_metrics[5]`）
**汇总**：`python tools/report_seed100.py`（逐 seed 可展开）
**事故**：08-17 08:18 同节点 `pkill -f` 误杀 2 块（回收重排）；退役 worker 多认领的 2 块回收；`retire_seed_worker.sh` 守卫解决竞态

**中期结果（08-18 05:24）**：

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

<a id="b8"></a>
## B8 · KIT x0-L1 同配方重训（进行中）

**Launcher**：`hml3d_x0_L_20260810.sh`（`DATASET=kit`）/ `hml3d_x0L_ep1000_20260814.sh`（1000ep）；tokenizer `vqvae_kit_partaware_datadriven_300k_seed3407_h200_20260722`（net_best_fid）；cache `kit_qwen3_l2v_normavg_v2_20260807`（6368/6368 覆盖）；其余与 HML3D L1 一字不差；gate 0.25/0.83

### B8a · 600ep `kit_x0L1_e1_nopool_20260816`（08-17 07:01→11:25，H100 1332151）
| 阶段 | 现象 |
|---|---|
| ep100–220 | 采样崩塌：FID 81→84.6，R@3 0.08–0.11（≈随机 3/32），Div 2.5–2.8（真实 11.1）；**cfg1 同崩**（诊断 `eval_results/kit_x0_diag_20260816/`）→ 非引导过强 |
| ep260–340 | 相变：FID 55.8→3.86，R@3 0.196→0.645 |
| ep340–600 | 单调改善：ep410 0.492/0.776 → ep500 0.181/0.805 → **ep600 0.1374/0.8111（best_fid）**；best_top3 ep530 0.1576/**0.8224**；末 6 点 FID 仍降 |
| 对照 v 时代（带 CE 的 kit_20260807 ep350） | 0.1791/0.8509（单次），ep200 已 0.93/0.815 |

**判读**：x0-flow-only 在 KIT 上**晚熟 ~200ep**（训练侧 loss/token_acc 全程正常，是采样输出成型晚）；FID 已反超、R@3 差 0.03；有效训练只有起飞后 ~300ep → 处方是拉长而非改配方。我曾误判为"坏了"，已收回。

### B8b · 1000ep `kit_x0L1_ep1000_20260818`（08-18 06:0x 发车，H100 1332152，与 seed worker 共卡）
- 配置实锤（options.json）：kit / lr 1e-4 / max_epoch 1000 / uniform / t_eps 1e-4 / llm2vec_cache + modulation / seed 42；ep0 loss 40.35 与 600ep 版逐位一致
- 待补：ep800+ 退火段结果；达标线 ≈ 0.17/0.85

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
