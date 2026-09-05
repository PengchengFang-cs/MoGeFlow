# MoGeFlow 实验计划（回溯整理版，持续更新）

> 按 ARIS `/experiment-bridge` 的 `EXPERIMENT_PLAN` 模板整理。**这不是从零起草的计划，而是把已经跑过、正在跑、还要跑的实验按 claim → 实验块 → 里程碑的结构对齐**，让任何一次新会话都能在 5 分钟内接上项目。
> **与旧记录的关系**：本目录里带 `_20260610_170154` 时间戳的 PLAN/TRACKER 属于 6 月的"统一 T2M + inpainting"支线（已搁置，见 `PROJECT_DIRECTION_LOCK.md` 硬边界：只做基础 T2M）；`refine-logs/MANIFEST.md`、`round-*`、`FINAL_PROPOSAL*` 属于 5–7 月的 research-refine 运行。固定名文件从 2026-08-18 起指向当前主线（MoGeFlow x0 时代）。5–7 月的实验永久记录在根目录 `CODEFLOW_EXPERIMENT_LOG.md`（05-21→07-23）与 `NARRATIVE_REPORT.md`（07-12）。
> 姊妹文件：`refine-logs/EXPERIMENT_TRACKER.md`（执行清单：什么在跑/待跑）、`EXPERIMENT_LOG.md`（永久记录：跑了什么、结果如何、怎么复现）、`MANIFEST.md`（产物索引）。
> 最后更新：2026-08-20（论文定稿口径：保持 v 时代数字；x0 结果转为后续依据）

**问题**：文本到人体动作生成里，离散 token 接口（分类式 masked/AR 模型）把生成限制在码本格点上；连续流模型又丢掉了 Part-VQ 码本的结构先验。能否在**单层 Part-VQ 码本的连续嵌入空间**上做流匹配，既保留部件结构、又不受格点约束？

**方法论点**：MoGeFlow —— 在 Part-VQ（6 部件、单层、K^P 积格）码本嵌入空间上训练 x0-预测流匹配（损失在速度空间计算），终端状态直接连续解码（不重量化）；文本条件走 LLM2Vec-Qwen3 token 序列的联合注意力，AdaLN 只带时间步。

**目标会议**：ICLR 2027（论文：`projects/Umdd/overleaf-paper/iclr2027_conference.tex`）

---

## Claim 映射

| Claim | 重要性 | 最低说服力证据 | 关联实验块 | 状态 |
|---|---|---|---|---|
| C1 主结果：HumanML3D 上 R@3 达到/超过 SALAD 级别（≥0.86）且 FID 处于领先段位 | 论文主表 | 一个检查点、一个 cfg、多 seed 均值±CI 的 Table 1 行 | B1, B4, B7 | **进行中**：x0L1 best_fid，cfg 9/10 各 100 seed（08-17→） |
| C2 KIT 上不掉队（检索领先、FID 可竞争） | 主表第二数据集 | 同 C1 协议的 KIT 行 | B8 | **进行中**：KIT x0 600ep 落后 v 时代 R@3 0.03；1000ep 08-18 发车 |
| C3 几何诊断：码本距离与解码器响应相关（ρ≈0.82，白化后 0.84），且诊断有区分力（通用 RVQ 只有基层强） | 方法动机 | 相关→打乱对照→因果替换 + 白化度量 + RVQ 对照 | B0（已完成，v 时代，与预测头无关） | ✅ 完成（08-09） |
| C4 连续直接解码优于终端投影（nearest） | 贡献 3 | 同检查点 continuous vs nearest repeat-20 | B0 | ✅ 完成（07-30，0.0577 vs 0.0686，CI 不重叠）；nearest 已从代码删除 |
| C5 文本编码器/路由选择：LLM2Vec-Qwen3 nopool 是最稳的臂 | 消融表 | 5 臂同协议 600ep + cfg 全扫 | B2, B3 | ✅ 完成（08-10→08-13） |
| C6 训练配方在 x0 时代已被钉死（t 采样、cfg 区间、lr、长度） | 复现性 / 审稿人问 | 每个替代方案一次同协议对照 | B5, B6 | ✅ 全部负结果（08-13→08-16） |

---

## 实验块

### 实验块 0（B0）：v 时代基线与几何诊断（2026-05 → 08-09，历史）
- **状态**：已完成并归档；模型全部为 velocity 头。**用户 08-18 澄清：旧检查点与其评测数字并未作废**——只是不能在 x0 代码下 resume/复用权重；**用户 08-20 最终裁定：论文全文保持 v 时代数字不动**（本来就自洽）；x0 时代的全部结果作为后续工作的方向依据，不进本篇。此前『只重填 Table 1 两行』的计划作废。
- **内容**：Part-VQ tokenizer 选定（`new_vq_overlap_top3_20260529`）；MoGeFlow-L 主线；文本编码器 6-run 消融（CLIP-L / LLM2Vec-Llama3 / LLM2Vec-Qwen3 × pool）；pooled-routing 6-run；v 时代 cfg 扫描 72/72；repeat-20 1/8；HML3D/KIT gate 重训（**事故**：terminal CE 权重意外为 1.0，见 `memory/flow-only-loss-permanent.md`）；KIT ep350 repeat-20 进论文；几何诊断 E1/E2/E3
- **记录**：`docs/TEXT_CONDITIONING_EXPERIMENTS_20260731.md`、`docs/ENCODER_ABLATION_FINAL_20260802.md`、`docs/POOLED_ROUTING_CAMPAIGN_20260802.md`、`docs/PAPER_REVISION_CAMPAIGN_20260808.md`
- **对后续的约束**：一行=一个检查点（禁止跨检查点拼接）；FID 单次抖动 ~0.01，排名必须用多 seed；论文数字"一个模型一个数字"

### 实验块 1（B1）：x0 硬编码 + 单一速度空间损失（2026-08-09/10）
- **验证 Claim**：C1 的前提（用户已在 MotionCraft 私有仓库验证 x0 预测更好）
- **改动**：`PREDICTION_TYPE="x0"`；loss = MSE(velocity_from_clean(z_t,t,x0_pred), Y1−Y0)；t∈(1e-4,1−1e-4)；CFG 在 x0 空间合成后转速度；terminal/codebook CE + clean loss **删除**（不是默认关）；提交 62139ad → 8b78fac → 064a97d（分支 `x0-dev`，本地，永不 push）
- **验证**：数学一致性（CFG 等价 3e-6、末步落 x̂0 3.6e-7）、微过拟合（loss 0.0016，采样复现 4e-4）通过
- **优先级**：必须 ✅

### 实验块 2（B2）：x0-L 五臂编码器/路由消融（08-10 → 08-12）
- **验证 Claim**：C5
- **对比系统**：L1 LLM2Vec-Qwen3 nopool / L2 LLM2Vec pooltoken / L7 raw-Qwen3-normavg nopool / L6 raw-Qwen3 tok+refiner2（2×A100 DDP）/ L8 CLIP-B dual（token→context + pooled→AdaLN）
- **设置**：MoGeFlow-L（part 192 / hidden 1152 / 6+12）、bz 64、lr 1e-4（0.8/0.9/0.95 减半）、600ep、seed 42、EMA 评测、每 10ep 自 ep100 评测（cfg 6 / 96 步 / seed 42 / test / bz 32 单卡）；gate 保存阈值 0.08/0.87
- **成功标准**：两指标同时好（"不能只看 FID 或 Top3"）
- **结果**：L1 最稳（12 个平衡点）；L7 FID 最低；L8 垫底；gate 零触发 → 见 `EXPERIMENT_LOG.md#B2`
- **优先级**：必须 ✅

### 实验块 3（B3）：x0 检查点 CFG 全扫（08-12 → 08-13）
- **验证 Claim**：C5 / 选操作点
- **内容**：cfg {1.5,2,3,4,5,7,8,9,10,11}（cfg6 取训练期评测）× 5 臂 × {best_fid, best_top3} = 100 次单次评测（seed 42）；**用户要求全部曲线原样汇报，不许摘要压缩**
- **结论**：联合最优 cfg 9–10；R@3 峰 8–10，FID 至 11 仍缓降；x0L1 bf cfg9 = 0.0630/0.8733、cfg10 = 0.0555/0.8713 → 见 LOG#B3
- **优先级**：必须 ✅

### 实验块 4（B4）：操作点定稿 + 100-seed 统计（08-16 → 进行中）
- **验证 Claim**：C1
- **用户裁定（08-16）**：模型 **x0L1 best_fid（ep600）**；cfg **9 与 10 都做**；**每个 100 seed，逐 seed 记录**（seed 42–141）
- **实现**：5 seed/块 × 20 块/cfg，flock 队列，跨 H100/A100；每块 JSON 内 `repeat_metrics` 逐 seed 保留
- **状态**：08-18 05:24 cfg10 90/100、cfg9 10/100 → 见 TRACKER
- **优先级**：必须 🔄

### 实验块 5（B5）：训练配方挑战者（全部负结果，08-13 → 08-16）
- **验证 Claim**：C6
- **B5a JiT 对齐**（logit-normal(−0.8,0.8) t 采样 + 分母 clip 0.05；三臂 L1/L7/L2 600ep）：全面落后且后段走坏 ✗
- **B5b lr 2e-4 + 1000ep**（三臂；milestones 自动落 800/900/950）：FID 端全部不如 1e-4/600ep，联合平衡点 0 个 ✗（1e-4@1000ep 对照被用户取消以省卡）
- **裁定**：uniform t / t_eps 1e-4 / lr 1e-4 / 600ep 保留
- **优先级**：已完成 ✅（负结果）

### 实验块 6（B6）：采样侧提点——区间引导（08-14）
- **验证 Claim**：C6
- **内容**：`sample_embeddings` 增加 `cfg_t_lo/cfg_t_hi`（默认 0/1 与恒定引导逐位一致，冒烟测试通过）；x0L1 bf、s=10、11 个窗口
- **结论**：砍尾段（t>0.7）零影响；砍头段每 0.1 崩一档 → 引导收益几乎全在 t<0.1；恒定 cfg 保留 ✗
- **优先级**：已完成 ✅（负结果）

### 实验块 7（B7）：论文数字定稿 —— **取消（08-20）**
- **裁定**：论文保持 v 时代数字与 v 时代 Method，全文自洽，不做 x0 重填。08-20 曾按 cfg10 前 20 seed 填过 Table 1 / 摘要 / §4.2 四处，随即按用户指示 `git checkout` 全部复原，Overleaf 工作区干净（HEAD eb89474）。
- **理由（用户原话）**：全文保持以前的，本来就是自洽，最近跑的只是给未来一个指引
- **仍待用户自己处理**：AI use statement；bib-audit 提交（本地 eb89474，未 push）

### 实验块 8（B8）：KIT x0 同配方重训（08-17 → 进行中）
- **验证 Claim**：C2
- **B8a 600ep**（`kit_x0L1_e1_nopool_20260816`，H100 4h24m）：ep100–220 采样崩塌（FID 81、R@3 随机、Div 2.7）→ ep260–340 相变 → ep600 **0.1374/0.8111**（bf）、0.1576/**0.8224**（bt，ep530），FID 仍在降；v 时代 ep350 = 0.1791/0.8509
- **B8b 1000ep**（`kit_x0L1_ep1000_20260818`，08-18 06:0x 发车，H100 1332152 与 seed worker 共卡）：看 R@3 能否上 0.83–0.85
- **成功标准**：KIT 行两指标不弱于 v 时代（0.17/0.85）
- **失败解读**：KIT 小数据 + 纯 flow x0 需要更长训练；若 1000ep 仍不到，需在 KIT 上重新考虑训练长度/warmup（不改配方）
- **优先级**：必须 🔄

### 可选实验块（未启动）
- **B9 采样步数消融**（8/16/32/64/96，纯 eval，几小时）— flow 论文标配表；用户未拍板
- **MotionMillion 不重训**（用户 08-18：需 8×A100 约 15 天），论文沿用旧结果那一行
- 已明确**不做**：seed 研究以外的 CI、参数量对齐的分类式对照（E7）、MotionMillion CI、bz/dropout 调参、Heun 采样器、self-conditioning（架构改动）

---

## 运行顺序（实际时间线）

| 里程碑 | 目标 | 运行内容 | 决策关卡 | 实际耗时 | 状态 |
|---|---|---|---|---|---|
| M0（05→07月） | tokenizer + 主线 + v 时代消融 | B0 | 论文初稿 | ~2.5 月 | ✅ |
| M1（08-08→09） | 论文修订闭环 + 几何诊断 | B0 后半 | 3 审稿人 4→? | 2 天 | ✅ |
| M2（08-09→10） | x0 硬编码 | B1 | 数学/微过拟合通过 | 1 天 | ✅ |
| M3（08-10→12） | 五臂重训 | B2 | 谁最稳 | ~30h/臂 | ✅ |
| M4（08-12→13） | cfg 全扫 | B3 | 峰值被夹住 | 100 次×31min | ✅ |
| M5（08-13→16） | 配方挑战者 | B5, B6 | 全负 | 3 训练×2 + 6 训练×2 天 | ✅ |
| M6（08-16→） | 定稿统计 + KIT | B4, B8 | 100 seed 收齐；KIT 1000ep | ~2 天 | 🔄 |
| M7 | 论文数字与 Method 改写 | B7 | 用户拍板 | — | ⏳ |

## 算力预算与实际

- **卡池**：Iridis-X swarm_h100 / swarm_a100 / h200，用户账号下多作业；**1332155/1332156 等常被 xiabao（NumPro）占用，不碰**；用户其他训练（ISLES、mambarecon 等）不抢
- **单位成本**：HML3D-L 600ep ≈ 24–28h（H100）；KIT 600ep ≈ 4.4h（H100）；一次 test 评测 ≈ 31 min（H100）/ 70 min（A100）
- **B4 总量**：200 次评测 ≈ 100 H100 小时
- **规则**：跳板机 CPU 禁令（全局 CLAUDE.md）—— 一切计算走 `srun --overlap` 到已分配作业

## 风险

- **KIT 在 x0 配方下晚熟**（相变 ep260–340）→ 缓解：1000ep（B8b）；若仍不足，考虑 KIT 专属更长训练而非改配方
- **同节点多作业误杀**（08-17 `pkill -f` 误杀两个 seed 块）→ 只按 PID 杀；重排即可
- **会话/tmux 重启会带走 srun 训练步**（08-14 三条 lr2e4 训练死亡，从 latest.pt 无损续跑）→ 每次训练存活判据 = GPU 占用 + 日志新鲜度，不看 epoch 数字
- **单 seed 乐观偏差**：seed 42 在 100 seed 里偏好端（cfg10：0.0555 vs 均值 0.0678）→ 论文只用多 seed 均值
- ~~论文 KIT 行待替换~~ → 08-20 取消：论文全文保持 v 时代数字

## 决策登记（已定案，不再重提）
- **论文表述定案（08-20）**：全文按 CLIP ViT-B/32 描述条件路径（tokens 进联合注意力 + pooled 进 AdaLN）；KIT 行用 gate_ep0960 @ cfg5 的六个指标；± 沿用既有值不再重测；不写任何 epoch 限定语与防御性表述；配置只在实现附录澄清一次
- **本篇论文使用 v 时代数字，全文不改**（08-20）；x0 时代结果 = 后续方向依据
- 单一损失（x0 头、速度空间 MSE），terminal/codebook/clean 损失永久删除
- 一个 run 只许一个文本编码器；continuous 解码焊死
- 一行=一个检查点；报 best-FID 需附同点 R@3、报 best-R@3 需附同点 FID
- 论文：一个模型一个数字；不做防御性写作；split 只说"on HumanML3D"；MotionMillion 只报一条标准行；cfg 不进论文；引用不擅动
- 不 push、不跑 review/repeat 等额外流程，除非当次明确指令
- 报告只报 best 检查点 + 进度，不报最新 epoch 快照
- **guidance 扫描不进正文（08-30）**：审稿人要求的 guidance-scale Pareto 图不放主表/正文；数据已有（B3 HML3D 100 点、B8c KIT 32 点），如需回应 rebuttal 只以附录形式给出。理由：本领域主表惯例是各方法在自己作者选定的采样超参下发布，主表内附扫描不是惯例
- **审稿 agent 的领域校准（08-30）**：给审稿 prompt 补 Guo et al. 2022 评测协议背景（冻结评测器、Real 行是参考行而非天花板、Diversity 是 → 列）与"判定数字异常前先扫整列"的硬性规则。首轮三人独立把"R-Precision 超过 real"当作缺陷，而 Table 1 中 11 个基线里 9 个同样高于 real（SALAD +0.060、MotionHiFlow +0.046，我们 +0.077）——论文自己的表就能否定该批评
- **Discrete Diffusion 不做参数配平（08-31）**：跨模型族"配平参数"没有可定义的口径，且该臂已训过。问题不在实验，在 L715 把 "parameter budget fixed" 写成整张 Table 4 的统摄句。改法：该句限定到两条 generic RVQ 行（677M/694M vs 690M，本来就配平），A.4 补一句该臂的描述与引用。不跑任何实验
- **划分类消融一律引 KV-Control（08-31）**：P/K 消融、统计划分 vs 手工划分、P=1 整身 VQ 全部不做——tokenizer 不是本文贡献，已在 KV-Control 做过。代价：L716「applying it over this lattice is [the active ingredient]」与 L717 的归因必须降级到 Table 4 实际测到的范围
- **ρ↔下游剂量–反应不做（08-31）**：改为把 L98 / L640 的 "licenses" 降级为 "motivates"，撤掉"测量授权了设计"这条因果承诺。理由：附录自己给出 generic RVQ base ρ=0.80 ≈ PartVQ 0.822，而下游 0.854/0.094 vs 0.874/0.048——ρ 在唯一一次可检验的对比里没有区分力；不改词就欠一个预测性实验
- **Table 3(b) 统一到 Table 1/4（08-31，作者裁定）**：codebook 参照行改为 L `0.874/0.048`、B `0.864/0.054`（即主表同一检查点）；encoder-latent / variational 两臂保持不动（各自 ~ep250，是它们自己的 best-Top3）。口径统一为"每条臂报自己的最优检查点"。连带：Intro L99 删掉 `and the training budget`；A.4 L833 改写为"固定 flow 配方 / 冻结 tokenizer / 评测协议，每条臂报自己最优检查点"
- **decode-mode 两行都带 ±（08-31，作者裁定）**：`0.873±.002 / 0.058±.003` 与 `0.874±.002 / 0.048±.003`。留痕：nearest 那行在 docs/ 中查无 repeat-20 实测记录，作者以"同一模型同一协议"裁定照写；nearest 解码已从代码删除，无法重跑
- **Discrete Diffusion 行按作者陈述写（08-31）**：作者在本目录早期设计阶段跑过，是 MoMask 族的 masked categorical 先验，同一 part-structured 格点。已写入 A.4。**不加 † early-experiment 脚注**——08-08 campaign doc 里那条是我们自己的会议纪要，不是作者指示，且无依据
- **licenses → motivates（08-31）**：L98、L640 已改，撤掉"测量授权设计"的因果承诺
- **L715 限定到 RVQ 两行（08-31）**：改为"匹配参数预算（677M/694M vs 690M）+ 同一 flow 配方"，删掉 `changing only the codebook interface`。A.4 补一句 generic RVQ 容量严格更大（6 层×512×512 vs 单层×128×128）、重建上限更高，因此"重建更差"解释不成立——**以此替代 J4 重建实验（作者裁定不跑）**
- **B9 三条结果入论文（08-31）**：(1) Table 3(a) 增设 `ρ decoded` 列（六组 0.962/0.964/0.603/0.942/0.595/0.938，均值 **0.834**），caption 与 §3.2 原型定义相应扩写，§4.3 报出 0.834 对 0.822；(2) A.3 未训练对照段追加配平支撑句（37 码下 0.820 对全支撑 0.822）；(3) A.3 新增 `Off-lattice offset directions` 段（in-span 0.220 对随机 0.063，3.5 倍，分组 2.7–4.7 倍），§3.4 与结论相应改写。编译 0 错误/0 溢出/0 未定义引用，正文仍 9 页多（REFERENCES 在 p.10），附录 14→15 页
- **非格点目标必须用自身统计量白化（09-05）**：B11 证实 encoder-latent 臂原先用码本统计量白化，制造了"目标空间几何更差"的假象（FID 0.151 实为 0.082）。任何新增的目标空间对照臂一律用其自身逐维统计量
- **guidance 扫描不进论文（09-04 再确认）**：论文所报的点本就在 Top3 与 FID 两项同时优于 SALAD，扫描只能证明"没挑幸运点"，属替未提出的质疑辩护。数据留 `eval_results/cfgsweep_june_20260904/`
- **v 时代检查点评测须带 `--head_is_velocity`（09-02）**：否则静默塌成噪声，不报错
