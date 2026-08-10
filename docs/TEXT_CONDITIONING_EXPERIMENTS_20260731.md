# MoGeFlow 文本条件实验记录

> ## ⚠ 本文件第 5 节的结论已作废，见 `ENCODER_ABLATION_FINAL_20260802.md`
>
> §5.1 表中的 ④'⑤'⑦'⑧ 四行，以及 §5.3 的三条结论——「全局分支有益」「refiner 有害」
> 「Qwen3 底座优于 Llama3」——全部建立在被污染的 LLM2Vec-Llama3 特征上（分隔符进
> tokenizer + pooled 池化范围错，见 §6.1），对应的检查点已删除，结论已撤回。
>
> §5.1 表的 best-Top3 列只写了 Top3、未附同一检查点的 FID，这种形式会诱导跨检查点拼接读法，
> 不要照此制表。最终结论与正确制表方式见 `ENCODER_ABLATION_FINAL_20260802.md`。
>
> 本文件仍然有效的部分：§3（编码器资产与 Qwen3-LLM2Vec 自实现）、§6（踩过的坑）、
> §9（配方对齐与 refiner 修复）。


最后更新：2026-08-01 00:30 BST（第 9 节：配方对齐与 refiner 修复）

本文档记录 2026-07-28 起进行的一轮实验，目标是回答一个问题：**哪种文本条件方案对
MoGeFlow 的下游生成质量最好**。涵盖实验设计、全部代码改动、当前结果、以及过程中
踩到的坑（含若干"看起来成功其实没生效"的静默失败）。

---

## 1. 背景与动机

### 1.1 起因

论文主表（`overleaf-paper/iclr2026_conference.tex`, `tab:main_results`）中 MoGeFlow
的 HumanML3D 行来自 6 月的主线运行，用的是单 CLIP ViT-B/32。这一轮想搞清楚：换更强的
文本编码器能否提升下游指标，尤其是我们最弱的 **R-Precision（Top3）**。

### 1.2 repeat-20 复核（2026-07-30）

填表前先按论文声明的协议复核了主表数字，发现两个问题：

1. **仓库里没有任何 repeat-20 的评测**——`eval_results/` 的 23 个结果、所有训练中评测、
   KIT 的 4 个 continuous 评测，`repeat_times` 全是 1。而表格 caption 写的是
   "± 为 repeat-20 协议下 95% 置信区间的半宽"。
2. 主表 HumanML3D 行（0.589/0.784/0.874/0.048/2.600/9.914）与 20260614 那次
   **repeat=1** 的结果逐位一致，其 ± 值无可追溯来源。

补跑 repeat-20（H200，steps 96 / CFG 6.0 / seed 42 / `--disable_mm` / test split /
4646 样本）：

| decode | Top1 | Top2 | Top3 | FID | MMDist | Diversity |
|---|---|---|---|---|---|---|
| continuous | 0.5829±.0035 | 0.7751±.0033 | **0.8608±.0025** | **0.0577±.0027** | 2.6144±.0061 | 9.7420±.0662 |
| nearest | 0.5837±.0035 | 0.7750±.0030 | 0.8615±.0021 | 0.0686±.0027 | 2.6141±.0062 | 9.7979±.0658 |

**检查点身份已验证无误**：repeat 0 使用 seed 42，逐位复现了 6 月 repeat=1 的数字
（FID 0.047862 / Top3 0.873922）。但在 20 次抽样中，seed 42 恰好是 **FID 最低的一次**。
所以论文的 0.048/0.874 是一次幸运抽样，诚实值是 **FID 0.0577±0.0027 / Top3 0.8608±0.0025**。

对论文的两处影响：

- 头条主张需改写。"把 Top-3 从 SALAD 的 0.857 提到 0.874"实际是 **0.861±0.003**，
  优势从 1.7 个点缩到 0.4 个点，约等于一个置信区间半宽。
- **贡献 3（直接解码优于终端投影）反而被坐实**：continuous FID 0.0577±.0027 vs
  nearest 0.0686±.0027，区间不重叠；而 Top1/Top2/Top3/MMDist 在 CI 内完全一致。
  准确表述应为"直接解码改善分布保真度，对检索对齐无影响"。

> 用户决定：论文暂不改，先把实验做完。

---

## 2. 实验设计

### 2.1 固定配置（所有 HumanML3D 运行一致）

| 项 | 值 |
|---|---|
| 模型 | MoGeFlow-L：`part_hidden_dim=192`, `hidden_size=1152`, 约 690M |
| 结构 | `depth_double=6`, `depth_single=12`, `num_heads=12`, `mlp_ratio=4.0`, `dropout=0.05` |
| 表示 | `part_structured` + `frame_grouped`，`code_dim=128`, `num_parts=6`, `num_codes=128` |
| VQ | HF `new_vq_overlap_top3_20260529_best_top3.pth` |
| 优化 | batch 64, lr 1e-4, half_cosine, warmup 2000, wd 0.01, grad_clip 1.0, bf16 |
| 训练 | 600 epoch, seed 42, `cond_drop_prob=0.1`, 无 self-conditioning |
| 损失 | flow-only（`terminal_loss_weight=0`, `clean_loss_weight=0`），`tied_logits` + `codebook_nn` |
| 评测 | 每 25 epoch 在 **test** 上评一次（从 ep0 起），steps 96 / CFG 6.0 / repeat 1 / seed 42 |
| 解码 | HumanML3D `continuous`；KIT `nearest`（与 6 月 KIT 基线对齐） |

### 2.2 变量矩阵

四个变量，每个都有干净的单变量对照：

| 变量 | 取值 | 对照 |
|---|---|---|
| **文本编码器** | CLIP-B/32、CLIP-L/14、Qwen 36层LN均值、LLM2Vec-Llama3-sup、LLM2Vec-Llama3-mntp、LLM2Vec-Qwen3-sup | ①vs⑥vs③vs④'vs⑧vs⑨ |
| **全局分支**（pooled 进 AdaLN） | 有 / 无 | ⑤'vs④'；⑩vs⑨ |
| **text refiner 深度** | 0 / 2 | ⑦'vs④' |
| **LLM2Vec 配方深度** | mntp-only / mntp+supervised | ⑧vs④' |

### 2.3 运行清单

| # | 文本条件 | 全局 | refiner | 备注 |
|---|---|---|---|---|
| ① | 单 CLIP ViT-B/32（tokens+pooled） | 有 | 0 | 6 月主线，论文数字来源 |
| ② | CLIP-L pooled + Qwen 末层 tokens | 有 | 2 | 7 月 10 日旧 run，停在 502/600 且**无法续训**（latest.pt 已不存在） |
| ②' | 同 ② 重跑 | 有 | 2 | 为进入最终对比表 |
| ③ | Qwen3 36 层 LayerNorm 均值，token-only | 无 | 2 | 7 月 28 日 |
| ④' | LLM2Vec-Llama3 **supervised**，token-only | 无 | 2 | LLM2Vec 主候选 |
| ⑤' | 同 ④' + pooled 进 AdaLN | 有 | 2 | 全局分支对照 |
| ⑥ | 单 CLIP-L/14（tokens+pooled） | 有 | 0 | 与 ① 同构，仅换编码器规模 |
| ⑦' | 同 ④' 但 refiner=0 | 无 | 0 | refiner 对照 |
| ⑧ | LLM2Vec-Llama3 **mntp-only**，token-only | 无 | 2 | 配方深度对照 |
| ⑨ | LLM2Vec-**Qwen3** supervised，token-only | 无 | 2 | 底座对照 |
| ⑩ | 同 ⑨ + pooled | 有 | 2 | Qwen3 上的全局分支对照 |

---

## 3. 编码器资产

### 3.1 CLIP 系

| 编码器 | 文本塔参数 | hidden | layers | heads | 来源 |
|---|---|---|---|---|---|
| ViT-B/32 | 63.4M | 512 | 12 | 8 | OpenAI，`KV-Control/checkpoints/clip/ViT-B-32.pt` |
| CLIP-L/14 | 123.1M | 768 | 12 | 12 | HF `clip-vit-large-patch14` |

### 3.2 LLM2Vec 系

LLM2Vec 是**配方**不是模型，三步：

1. **双向改造**——去掉 causal mask（单独做会掉点）；
2. **MNTP**——掩码预测，让模型学会用右侧上下文；
3. **对比学习**（二选一）：`unsup-simcse`（无标注，dropout 造正样本）或
   `supervised`（E5 检索三元组，约 150 万条）。

官方检查点：

| 底座 | 可用变体 | 授权 |
|---|---|---|
| Llama-3-8B-Instruct | mntp / mntp-supervised | 底座 gated（我们的 token 已获批） |
| Mistral-7B-v0.2 | mntp / unsup-simcse / supervised | 全开放 |
| Qwen2.5（0.5B/1.5B/3B/7B） | mntp / unsup-simcse / supervised | 全开放 |
| **Qwen3** | **不存在任何变体** | — |

用户要求必须用 Qwen3（"Qwen3 和 Qwen2.5 能力天差地别"），因此**自行实现并训练**。

### 3.3 自训 Qwen3-LLM2Vec

**版本死锁**：官方 `llm2vec` 包锁 `transformers<=4.44.2`（继承了 4.48 后被删除的
`LlamaFlashAttention2` 等类），而 Qwen3 模型代码需要 `>=4.51`。两者不可共存，
所以**移植方法而非复用代码**。

实现（`tools/llm2vec_qwen3/`）：

- `bidirectional_qwen3.py`：新版 transformers 把掩码收敛到
  `Qwen3Model._update_causal_mask` 一处，只覆盖它即可，无需继承注意力类。
  封堵两条静默退回因果的路径：SDPA 在掩码为 `None` 时自算
  `is_causal = q_len>1 and mask is None`（故掩码**永不返回 None**）；
  flash-attention-2 走独立逻辑（**直接拒绝该后端**，并把 36 个模块的
  `is_causal` 置 False 作为第二道防线）。
- `run_mntp_qwen3.py`：复刻官方语义——掩码 token 用 `"_"`（`blank`，复用词表 id 62，
  不扩词表）、20% 非特殊 token **100% 替换**（非 BERT 的 80/10/10）、
  "next token" 来自 CausalLM 内部的标签位移、LoRA r=16 / **alpha=2r=32** /
  dropout 0.05 挂在 7 个线性层、**lm_head 全量可训练**（上游从不冻结它）。
- `supervised_encoder.py`：复刻 `LLM2Vec` 的 tokenize/embed_mask/池化。
- `run_supervised_qwen3.py` + `vendored/`（E5 数据加载器、14 个数据集提示词、
  HardNegativeNLLLoss 原样复用 361 行）。

训练结果：

| 阶段 | 数据 | 步数 | loss | 产物 |
|---|---|---|---|---|
| ② MNTP | wikitext-103（241,754 个 512-token 块） | 1000 | 8.4669 → **1.4016** | 1.4G（LoRA + lm_head） |
| ③ supervised | echo-data / E5（150 万三元组，3.7G） | 1000 | 3.3024 → **0.2389** | 182M（LoRA） |

**编码器几何验证**（12 组同义句对，去公共方向后）：

| 编码器 | margin | 严格检索 |
|---|---|---|
| Qwen3 + MNTP | +0.2712 | 4/12 |
| **Qwen3 + supervised** | **+0.6981** | **12/12** |

第③步把严格检索从 4/12 拉到全对，证明自实现的管线有效。训练后双向性 **1.1387**
（训练前 0.90，改造未被破坏）。

---

## 4. 代码改动清单

### 4.1 文本 provider（`models/codeflow/text_encoder.py`）

| provider | 说明 |
|---|---|
| `clip`（既有） | OpenAI ViT-B/32，tokens(77×512) + pooled(512) |
| **`clip_large`**（新） | HF CLIP-L/14，tokens(77×768) + pooled(768，`pooler_output` 未投影） |
| **`dual_clip`**（新） | ViT-B/32 + CLIP-L/14。因 `text_encoder.*` 参数不进优化器也不存 checkpoint，provider 必须无参数：token 序列沿时间拼接（77+77），特征维分块对角放置（B/32→[0:512]，L→[512:1280]），使单个 `Linear(1280,hidden)` 等价于两个独立逐塔投影 |
| `clip_qwen_cache`（既有） | CLIP-L pooled + Qwen3 末层 tokens |
| **`qwen_normavg_cache`**（新） | Qwen3 36 层无参 LayerNorm 均值，token-only |
| **`llm2vec_cache`**（新） | LLM2Vec 逐 token 特征，`pooled=None` |
| **`llm2vec_pooled_cache`**（新） | 同上 + 池化句向量进 AdaLN |

`TextCondition.pooled` 改为 `Optional`；两个模型仅在 provider 暴露非空 `output_dim`
时才构建 `text_pooled_proj`，`cond = timestep_cond`（有则加 pooled）。**模型主体代码
未改**，两种形态靠同一套接口。

### 4.2 缓存构建器

- `tools/precompute_qwen_normavg_text_cache.py`（新）
- `tools/precompute_llm2vec_text_cache.py`（新，Llama）
- `tools/llm2vec_qwen3/precompute_qwen3_llm2vec_cache.py`（新，Qwen3，同一缓存格式）

一份缓存同时存**逐 token 特征**和**池化句向量**，两个变体共用，不必建两次。

### 4.3 检查点保留策略

`best_checkpoint_limit` 默认 **5 → 1**（`trainer.py` 常量 + options），并同步 22 个
启动脚本。今后每个训练只产生 `best_fid.pt` + `best_top3.pt` + `latest.pt`，不再有
`_rank2/_rank3`。

### 4.4 评测工具修复

`tools/eval_kv_part_vq.py`：失效的 `EVAL_SPLIT` 导入改为本地常量并固定 `test`；
`torch.load` 加 `weights_only=False` 适配 PyTorch 2.6。

### 4.5 GPU 调度器

`run_logs/launch/hml3d_ablation_scheduler_20260731.sh`：算力与用户共享，故不预先绑卡。
每轮重扫 `squeue`（新分配的 allocation 自动纳入）、排除用户保留的 allocation、
过滤剩余时长 <24h 的、只在显存 <1000MiB 时认定空闲，并用**持久化登记表**
（`status/hml3d_ablation_gpu_claims.txt`）防止重复派发。

---

## 5. 当前结果（2026-07-31 18:00）

### 5.1 HumanML3D 全局表

| # | 文本条件 | 全局 | refi | 进度 | best-FID | best-Top3 |
|---|---|---|---|---|---|---|
| ① | 单 CLIP ViT-B/32 | 有 | 0 | ✅600 | ep300: 0.0568 / T3 0.868 | ep290: 0.8731 |
| ② | CLIP-L pool+Qwen末层（旧） | 有 | 2 | 502 | ep460: 0.0517 / T3 0.863 | ep120: **0.8901** |
| ②' | 同上重跑 | 有 | 2 | 178 | ep175: 0.163 / T3 0.884 | ep150: 0.8849 |
| ③ | Qwen 36层LN均值 | 无 | 2 | ✅600 | ep425: **0.0407** / T3 0.860 | ep125: 0.8797 |
| ④' | LLM2Vec-Llama sup | 无 | 2 | 474 | ep450: 0.115 / T3 0.870 | ep325: 0.8752 |
| ⑤' | 同上 +pooled | 有 | 2 | ✅597 | ep550: 0.0582 / T3 0.863 | ep300: 0.8795 |
| ⑥ | 单 CLIP-L/14 | 有 | 0 | ✅600 | ep250: 0.0606 / T3 0.864 | ep150: 0.8780 |
| ⑦' | LLM2Vec sup refiner=0 | 无 | 0 | 449 | ep425: 0.0604 / T3 0.853 | ep175: 0.8765 |
| ⑧ | LLM2Vec mntp-only | 无 | 2 | 227 | ep225: 0.164 / T3 0.877 | ep125: 0.8784 |
| ⑨ | LLM2Vec-Qwen3 sup | 无 | 2 | 246 | ep225: 0.119 / T3 0.861 | ep175: 0.8705 |
| ⑩ | LLM2Vec-Qwen3 +pooled | 有 | 2 | 243 | ep225: 0.131 / T3 0.863 | ep100: 0.8647 |

### 5.2 同轮次 FID / Top3

| ep | ①CLIP-B | ③Qwen | ④'L2V | ⑤'+pool | ⑥CLIP-L | ⑦'r=0 | ⑧mntp | ⑨Qwen3 |
|---|---|---|---|---|---|---|---|---|
| 150 | 0.161/.869 | 0.185/.879 | 0.344/.861 | 0.355/.866 | **0.119**/.878 | 0.259/.876 | 0.258/.875 | 0.186/.864 |
| 200 | 0.107/.865 | 0.125/.873 | 0.312/.869 | 0.296/.873 | **0.080**/.869 | 0.182/.873 | 0.188/.877 | 0.145/.866 |
| 225 | — | 0.113/.871 | 0.287/.870 | 0.260/.879 | **0.068**/.868 | 0.158/.876 | 0.164/.877 | 0.119/.861 |
| 300 | **0.057**/.868 | 0.063/.868 | 0.217/.872 | 0.182/.880 | 0.069/.861 | 0.092/.863 | — | — |
| 425 | — | **0.041**/.860 | 0.122/.870 | 0.096/.861 | 0.100/.856 | 0.060/.853 | — | — |

### 5.3 四个变量的结论

**(1) 全局分支：有益，与预设相反。** ⑤' vs ④' 是唯一干净的单变量对照，⑤'
在 ep225/300/425 三个点上 FID 全面更优（0.260/0.182/0.096 vs 0.287/0.217/0.122），
Top3 也不落下风。⑤' 已跑满拿到 **0.0582**。此前基于 CLIP 得出的"全局梯度过大、
会让模型偷学"结论**不能外推到 LLM2Vec**。Qwen3 侧（⑩vs⑨）目前互有胜负。

**(2) refiner：去掉更好。** ⑦'（refiner=0）ep425 FID **0.060**，④'（refiner=2）
同期 0.122，好一倍。印证 `BidirectionalTextRefiner` 是为 **causal** 特征设计的补偿
模块，接在已双向的 LLM2Vec 特征后是冗余甚至有害的。代价是 Top3 略低（0.853 vs 0.870）。

**(3) 编码器：CLIP 仍领先。** ⑥ 单 CLIP-L 收敛最快（ep225 即 0.068），
①③⑥ 的 best-FID 都在 0.041–0.061，而 LLM2Vec 系需要更多 epoch 才追平。

**(4) 底座：Qwen3 优于 Llama3。** ep225 ⑨ 0.119 vs ④' 0.287，FID 好一倍多。
自训的 Qwen3-LLM2Vec 比官方 Llama3 版更适配本任务。

### 5.4 KIT 结果（已完成部分）

**tokenizer 重训**（test 选点，50k 步）：best-FID = step 10k（test FID 0.1326 / Top3 0.7656），
best-Top3 = step 16k（0.1580 / 0.7884）。此前用 val 选点的 155k 步检查点是 val 过拟合产物。

**下游 CodeFlow**（600 epoch，nearest decode）：

| 运行 | best-FID | best-Top3 |
|---|---|---|
| K1（best-FID tokenizer） | ep400: **0.2697** / T3 0.830 | ep275: **0.8395** |
| K2（best-Top3 tokenizer） | ep400: 0.3517 / T3 0.797 | ep300: 0.8224 |
| dual_clip（B/32 + L/14） | ep375: 0.3473 / T3 0.813 | ep300: 0.8310 |
| 6 月 formalvq 基线 | ep430: 0.4346 / T3 0.804 | ep290: 0.8352 |

K1 全面胜出，且大幅超过 6 月基线（FID −38%）。**dual CLIP 是负结果**，两个榜都输给单 CLIP。

**continuous decode 复评**（4 个检查点全部降 FID，7%–37%）：

| 检查点 | nearest FID | continuous FID |
|---|---|---|
| K1 best_fid | 0.2697 | **0.2506** |
| K1 best_top3 | 0.5057 | 0.3850 |
| K2 best_fid | 0.3517 | **0.2208** |
| K2 best_top3 | 0.5037 | 0.3962 |

"quantize to train, decode continuously" 在第二个数据集上得到支持。

### 5.5 HumanML3D 尺寸缩放（Qwen normavg，7 月 28 日）

| 尺寸 | best-FID | best-Top3 |
|---|---|---|
| W0.75（173M） | 0.0772 | 0.8720 |
| W1.0（307M） | 0.0538 | 0.8789 |
| W1.5（690M） | **0.0407** | **0.8797** |

两个榜严格单调，scaling 行为干净。

---

## 6. 踩过的坑

按严重程度排列。多数属于**静默失败**——不报错、照常训练、结论悄悄失真。

### 6.1 文本格式错误（最严重，导致 5 个训练作废）

官方 `LLM2Vec.encode()` 对每句都调 `prepare_for_tokenization`，包进 chat 模板；
`_convert_to_str("", text)` 还会前置分隔符 `!@#$%^&*()`，该分隔符驱动
`embed_mask`，使池化**只平均 caption token、排除模板标记**。

第一版 Llama 缓存喂的是**裸文本**，编码器处于分布外。发现后：停掉 5 个训练
（HumanML3D ×3、KIT ×2）、删除 383G 检查点和 15.6G 缓存、修正两个构建器、
新增 `verify_cache_text_format.py` 把链路固化成检查。

正确格式：

```
Qwen3 : <|im_start|>user\n!@#$%^&*(){caption}<|im_end|>
Llama3: <|start_header_id|>user<|end_header_id|>\n\n!@#$%^&*(){caption}<|eot_id|>
```

### 6.2 双向性测试无法分辨（bf16 噪声淹没信号）

第一版双向性检验中，**原版 causal 模型也"通过"**（比值 0.038 > 阈值 0.01）。
诊断发现是 bf16 数值噪声：同一模型 fp32 下只有 3e-6。改用 fp32 后 causal 基线 5e-6、
双向 0.90，相差 18 万倍。**若无对照组，会误判改造成功。**

### 6.3 左填充改变句向量（解释了参考仓库的 batch_size=1）

左填充使内容 token 的绝对位置后移，而普通前向的 `position_ids` 来自
`arange(seq_len)`、不看 attention mask，于是 RoPE 位置变了、向量随之改变（cos 0.963）。
**这正是参考仓库强制 `batch_size=1` 的机理**（它只说现象未说原因）。所有缓存构建
均采用 batch=1。

### 6.4 我的假设错误（三次）

1. 断言"lm_head 必须冻结"并触发——查证 upstream 从不碰 `lm_head`/`requires_grad`，
   **官方 MNTP 是训 lm_head 的**（Qwen3 为 622M 参数，`tie_word_embeddings=False`）。
2. 认为"跳过 instruction = 向量与 instruction 无关"——模型是双向的，内容 token 会
   注意到 instruction，**这正是把 instruction 放进输入的目的**。改用直接数值比对
   （内容均值误差 0.000000 vs 全 token 均值误差 7.47）。
3. 早期报告称 `run_supervised.py` 含 GradCache——实际没有，是凭印象误述。

另外 `lora_alpha` 一度写成 16，upstream 是 `2 * lora_r` = **32**，已修正。

### 6.5 lm_head 不属于 LoRA adapter

MNTP 训练 lm_head，但它不在 adapter 里，只存 adapter 会**静默丢弃**训好的 head。
已额外保存 `lm_head.pt`（1.19G）。

### 6.6 tmux 同名会话导致调度器重复派发

任务等缓存时不占显存，调度器重启后误判 GPU 空闲；更隐蔽的是
`tmux new-session` 遇同名会话**静默失败**，但登记表照写——结果 abl4 与 abl7 挤在同一块
H200（53G+55G）。已用持久化登记表 + 会话存在性检查修复。

### 6.7 杀进程导致 latest.pt 损坏

迁移 abl7 时杀掉进程，恰逢 `latest.pt` 写入中，续训报
`PytorchStreamReader failed reading zip archive`。改用完好的 `best_fid.pt`（ep125）
恢复。**同一故障此前在 H3 上出现过一次**（rose09 到期时）。

### 6.8 静默失败的补丁

- Python `str.replace` 因引号嵌套未匹配，**不报错、什么都不做**，导致防重复机制从未生效；
- grep 模式把 `$` 转义成字面美元符，占用检查永远不匹配。

教训：补丁后必须验证目标函数与调用点**都存在**，并实测行为。

### 6.9 环境与集群

- `conda run` **不转发 stdin**，heredoc 脚本被静默跳过（退出码仍为 0）。改用脚本文件。
- 计算节点上 `conda activate` 偶发失败并回退到默认 python。改用**绝对路径**调用解释器。
- 计算节点**无外网**，`datasets` 即使离线也要联网解析元数据。改为登录节点预处理 + `load_from_disk`。
- 登录节点在 **00:00 和 12:00** 清理用户进程，杀死 tmux 与 srun 客户端；Slurm 记为
  "CANCELLED by 56935"。跨越这两个时点的训练需从 latest.pt 续训。
- 集群管理员（root）曾于 2026-07-28 14:28/14:50 取消全部闲置 allocation 与整个排队队列。

---

## 7. 磁盘与资源

- 一轮清理共释放约 **646G**：作废训练 383G、错误缓存 15.6G、冗余 rank2/3 检查点 198G、
  已完成 run 的 latest.pt 49.5G。
- `checkpoints/` 由 574G 降至 327G。
- 权重资产：Llama-3-8B 15G + 两个适配器 330M；Qwen3-8B-mntp 1.4G；
  Qwen3-8B-mntp-supervised 182M。
- 缓存：HumanML3D 各约 7.5G（55,862 caption），KIT 各约 0.6G（6,368 caption）。

---

## 8. 后续待办

### 8.1 进行中

六个 HumanML3D 消融（②'④'⑦'⑧⑨⑩）与两个 KIT 终局（LLM2Vec-Llama sup、Qwen3 sup）
仍在训练，预计 1–2 天跑满。

### 8.2 待决策

1. **主表用哪个方案**。目前 FID 最优是 ③（0.0407），Top3 最优是 ②（0.8901，但该点
   FID 0.174），单点最均衡是 ①（0.058/0.873）与 ⑥（0.061/0.868）。需等 ④'⑤'⑦'⑨⑩ 跑满。
2. **是否补 `refiner=0` × 其他编码器**。目前只在 LLM2Vec 上测过 refiner，若结论是
   "refiner 只对 causal 特征有用"，则 ③（Qwen normavg，也是双向的？）也该测。
3. **repeat-20 正式评测包**。最终入表的检查点必须补 repeat-20，且 KIT 侧同样需要。
4. **② 的 502-epoch 旧 run 是否弃用**，改用 ②' 重跑结果。

### 8.3 尚未做

- **Qwen3 unsup-simcse**（第③步的另一口味）未实现，若要与 Mistral 那条无监督 SOTA 线
  对标则需补。
- **KIT 侧的 refiner / 全局分支消融**未做，目前只在 HumanML3D 上做了。
- 论文表格的更新（等实验定论）。

---

## 9. 配方对齐与 refiner 修复（2026-07-31 夜 ~ 08-01）

> **状态（08-01 01:20 更新）：代码已全部改完并通过回归测试；缓存 dry-run 已执行并通过
> （见 9.8）；仍未启动任何训练或缓存构建。**

### 9.1 起因

用户对 LLM2Vec 路线有信心（"我在同样我们模型架构的别的模型上训练效果也是显著的"），
要求复查实现是否有问题。复查发现我们的 Qwen3 配方与官方 LLM2Vec 有偏差，且 refiner
的池化把 chat 脚手架混进了全局摘要。

### 9.2 我们的 Qwen3 配方 vs 官方（已用磁盘上的上游仓库逐条核对）

上游仓库：`/scratch/pf2m24/projects/Umdd/external_repos/llm2vec`

| 项 | 我们原来 | 官方 | 出处 | 已对齐？ |
|---|---|---|---|---|
| MNTP 掩码方案 | 100% 换掩码 | **80% 掩码 / 10% 随机词 / 10% 原样** | `train_configs/mntp/MetaLlama3.json:12` = `"data_collator_type": "default"` | ✅ 改用 HF `DataCollatorForLanguageModeling` |
| MNTP warmup | 100 | **0**（配置未设 → HF 默认） | `train_configs/mntp/MetaLlama3.json` 无该键 | ✅ 改为 0 |
| supervised loss scale | 20 | **50** | `experiments/run_supervised.py:237` `loss_scale: float = field(default=50.0)`；supervised 配置未覆盖 | ✅ 改为 50 |
| supervised 负样本数 | 64（单卡） | 512（8 卡） | `per_device_train_batch_size: 64` × 8 GPU | ❌ **用户决定保持 64**（单卡，不做 GradCache） |
| 其余（lr 5e-5/2e-4、batch 32/64、seq 512、LoRA r16 α32 dropout 0.05、1000 步、supervised warmup 300） | — | — | — | ✅ 本来就一致 |

**关于 scale 50 的出处**：它来自实验脚本 `CustomArguments` 的默认值，**不是**
`HardNegativeNLLLoss` 类的默认值（那个是 **20**）。一份早期注释把两者搞混了，已改正。
这点重要，因为仓库里 `tools/llm2vec_qwen3/vendored/HardNegativeNLLLoss.py:8` 就写着 20。

**关于负样本数**：普通梯度累积**不能**增加 in-batch 负样本——累积 8 步 × batch 64，
对比损失仍只在那 64 句内部计算。要在单卡上真正得到 batch-512 的梯度需要 GradCache
（两遍前向 + 缓存 ∂L/∂e，梯度与 batch-512 逐位等价，约 12–16 小时）。用户评估后决定
**不做，batch 64 足够**。结论里应注明"负样本数为官方的 1/8"。

### 9.3 refiner 内容跨度池化

**问题**：`BidirectionalTextRefiner` 对**全部有效 token**取均值来构造 AdaLN 的全局
条件，而 chat 包装的 provider 把 `<|im_start|>user` 之类脚手架也存进了 token 流
（短 caption 里约占 26%）。均值是固定算子，学不会忽略它们；注意力可以。

**一个走不通的改法（已否决）**：让 refiner 直接用缓存里的 `pooled.npy`。
`CachedLLM2VecTextEncoder(use_pooled=False)` 返回 `pooled=None`，而 ⑨/④ 这些
token-only 变体正是靠"没有全局向量"成立的——喂 pooled 等于把全局向量塞回去，
毁掉这组消融。

**实际改法**：给 refiner 传一个**内容跨度掩码**，只窄化 pooled 均值；注意力块仍看到
全部有效 token。不引入任何全局向量。

- `TextCondition` 新增 `content_mask: Optional[Tensor] = None`
- 缓存新增 `spans.npy`：每条 caption 的**尾部**内容 token 数，内容位于 `[length-span, length)`
- 新参数 `--text_refiner_pool {content_span, all_tokens}`
- provider 无法给出跨度时 → **报错，绝不静默回退**

**只覆盖 LLM2Vec 系**。`qwen_normavg` / `clip_qwen` 用**右填充**且 caption 在序列
**中间**，"尾部跨度"表示不了，两者的 `_spans` 硬编码为 `None`（`provides_content_mask`
为 False）。→ 见 9.6 待决策。

### 9.4 两轮子 agent code review 发现的 11 个问题（全部已修）

**模型侧：**

1. **（严重）所有旧 checkpoint 会被用 `content_span` 评测。**
   我原以为"dataclass 默认 `all_tokens` = 向后兼容"，**这个推理是错的**：
   `eval_t2m_cli.py:114` 和 `gen_codeflow_t2m.py:154` 用 `parse_args([])` 拿 argparse
   默认值，然后**只覆盖 checkpoint 里存在的键**；旧 ckpt 没有这个键，argparse 默认值
   存活，dataclass 默认值根本轮不到用。后果：llm2vec ckpt 评测指标静默偏掉；
   `qwen_normavg`/`clip_qwen` ckpt 直接崩且 eval CLI 无参数可覆盖。
   → **修法：把 argparse 默认值翻成 `all_tokens`**，eval/gen 一行不改即自动全对
   （旧 ckpt 无键 → all_tokens；新 ckpt 有键 → 从 ckpt 覆盖）。
2. `train_humanml3d_pscf_qwen_normavg.sh` 会在分配到 GPU 后几分钟才崩 →
   随默认值修复，并新增**启动期预检** `_assert_refiner_pool_supported()`。
3. 续跑丢失 best 记录（`selection_configs_match` 缺垫片）；`load_checkpoint` 不校验
   pooling 模式，会静默换算子 → 均已加。
4. `_load_content_spans` 只验形状不验内容（两个 llm2vec 缓存 caption 列表相同 →
   形状也相同，`spans.npy` 拷错会静默生效）→ 加了与本缓存 `offsets` 的交叉校验。

**工具侧：**

5. `--loss_scale` 注释理由错误（见 9.2）→ 已改正。
6. resume 报错指向的恢复路径走不通：迁移工具拒绝 `building` 状态，而那正是 resume
   失败的场景 → 迁移工具改为接受 `building`（跨度只依赖 caption+tokenizer，
   两者在编码开始前就已定稿）。
7. `assert_resume_matches` 漏比 `model_dtype`（bf16 建一半用 fp16 续跑，前后半个
   缓存精度不同且全部有限）→ 已纳入。
8. **被截断的 caption 静默退回"对全序列取均值"**，即退回已修过的那个 bug
   （HumanML3D 56k 条里 1 条）→ 计数 + 显式 WARNING + 写入 manifest。
9. Qwen3 构建器每条 caption 跑**两遍** trunk → 改为从已算出的 hidden 池化；
   并在第 0 条对拍验证快捷路径**逐位相等**，不等就拒绝建缓存。
10. `--batch_size` 是空操作却被当事实写进 manifest 且进了 fingerprint 哈希 →
    限制为 `choices=[1]`。
11. `pooling` 常量仍是 `"masked_mean"`，导致**修复前后的缓存哈希相同**，
    pin fingerprint 分不出来 → 改为 `"masked_mean_over_content_span"`。

**另外修的（用户列表内）**：Qwen3 构建器的 `--resume` 原本是**假的**——打印
"pass --resume" 后用 `mode="w+"` 重建 memmap 从第 0 条重编码；已实现真正的
progress.json 续跑。双向性哨兵：导入期检查 `Qwen3Model.forward` 是否仍调用
`_update_causal_mask`（transformers 新版把它换成模块级 `create_causal_mask` 会让我们的
override 变成死代码，attention 静默退回因果）+ 运行期计数器 `assert_mask_override_fired`。

**未动**：timestep 缩放。它是 `3c99419 Initial portable training code release` 就有的，
不是这轮引入的；t∈[0,1] 喂正弦嵌入使有效秩为 2/256，但 t 本身一维、无信息损失，
且对所有变体一致，不构成跨 provider 比较的混淆。改它会让所有历史结果失去可比性。

### 9.5 已完成的验证

- 12 个文件在 `fudoki-momask` / `dllm` 两个环境下编译通过
- refiner 行为测试 **17/17**。注意：`AdaLNModulation` 和 `TextRefinerBlock.gate_modulation`
  **都是零初始化**，未训练模型上 `cond` 对输出完全无影响——最初两版测试因此是**空的**
  （断言恒真）。改用前向钩子直接抓 pooled 向量比对，并把门控打成非零后再测输出。
- checkpoint 恢复路径回归测试 **13/13**：模拟 `eval_t2m_cli` 的恢复流程，旧 ckpt 落回
  `all_tokens`、新 ckpt 保留 `content_span`。
- review agent 用**真实缓存实证**验证：跨度规则在 9 种 caption 形态 × 两个 tokenizer 上
  **零不匹配**；三个磁盘缓存长度与 offsets 全部吻合（40/40、6368/6368、55862/55862）；
  分隔符从未进过模型输入；哨兵在 transformers 4.52.4 上不会误触发；
  `torch.randint(len(tokenizer))` 不越界（151,669 < 151,936）。

### 9.6 待办与待决策

**待用户决策**：
- **refiner 修复只覆盖 LLM2Vec 系**。③（qwen_normavg）和 ②'（clip_qwen）同样有脚手架
  稀释，但留在 `all_tokens`。
  - 方案 A：接受，对比表注明"refiner 池化方式随 provider 不同"，不重跑任何东西
  - 方案 B：把跨度表示改成 (start, end) 通用形式，③②' 也修——但这两个要**重跑**
  - 倾向 A（③已完成、②'在跑，重跑代价大于收益）
- 是否重训 Qwen3 MNTP + supervised 以应用 9.2 的对齐（约 4–6 小时）。不重训也能用，
  但结论须写明"这是 LLM2Vec 风格实现，非配方复现"。

**已保护的在跑运行**：⑨⑩②' 正在训练，进程已加载旧代码不受磁盘改动影响。但**续跑会
读到新代码**，故已把 6 个 launch worker 的 `--text_refiner_pool` 钉死：已训练的 run →
`all_tokens`，待重跑的 → `content_span`。

### 9.7 新增/修改文件

```
新增  tools/add_content_spans_to_cache.py          给已有缓存补 spans.npy（纯 tokenizer，无 GPU）
改    models/codeflow/text_encoder.py              TextCondition.content_mask、_load_content_spans、
                                                   _content_mask_from_spans、provides_content_mask
改    models/codeflow/dit_blocks.py                BidirectionalTextRefiner 接受 content_mask
改    models/codeflow/motion_code_flow.py          text_refiner_pool 配置、_refiner_content_mask、
                                                   _assert_refiner_pool_supported
改    models/codeflow/part_structured_motion_code_flow.py   同上的调用侧
改    models/codeflow/trainer.py                   续跑校验 + best-metric 垫片 + 参数透传
改    options/codeflow_options.py                  --text_refiner_pool（默认 all_tokens，见 9.4-1）
改    tools/precompute_llm2vec_text_cache.py       spans.npy、resume 校验、截断告警、常量修正
改    tools/llm2vec_qwen3/precompute_qwen3_llm2vec_cache.py  真续跑、spans、单次前向
改    tools/llm2vec_qwen3/run_mntp_qwen3.py        80/10/10、warmup 0
改    tools/llm2vec_qwen3/run_supervised_qwen3.py  scale 50
改    tools/llm2vec_qwen3/bidirectional_qwen3.py   双向性哨兵
改    run_logs/launch/*.sh, scripts/launch/*.sh    钉死 --text_refiner_pool
```

### 9.8 dry-run 验证结果（2026-08-01 01:10，纯 CPU、未写任何文件）

`tools/add_content_spans_to_cache.py --dry_run`，dllm 环境（transformers 4.52.4）：

| 缓存 | n | 结果 |
|---|---|---|
| `humanml3d_qwen3_l2v_sup_l128_v1` | 55,862 | ✅ 通过，mean_span 16.55，内容占比 0.822（脚手架 17.8%）；1 条截断 caption 触发预期 WARNING |
| `kit_qwen3_l2v_sup_l128_v1` | 6,368 | ✅ 通过，mean_span 11.98，内容占比 0.777（脚手架 22.3%） |
| `_smoke_llama_fixed`（40 条，llm2vec 配方） | 40 | ✅ 通过，内容占比 0.673 —— Llama 分支的跨度推导同样成立 |

关键点：工具在写任何东西之前先用当前 tokenizer 重算长度并与缓存的 `offsets.npy` 逐条比对，
三个缓存**零不匹配**，说明跨度推导与缓存构建时的 tokenizer 一致。截断告警按 9.4-8 的设计
显式打印，未静默退回全序列均值。

**顺手修的一个缺陷**：`add_content_spans_to_cache.py` 的 docstring 声称拒绝
`momask_qwen3_normavg_cache`，但拒绝分支不可达——非 LLM2Vec 缓存的 manifest 根本没有
`encoder` 块，`manifest["encoder"]` 先抛 `KeyError`。改为 `manifest.get("encoder", {})`
并在缺失时回退到 `format` 字段。现在 `qwen_normavg` 与 `clip_qwen` 两个缓存都按名字被
干净拒绝（已实测）。

---

## 附录：关键路径

```
代码
  models/codeflow/text_encoder.py                     全部文本 provider
  tools/precompute_llm2vec_text_cache.py              Llama 缓存
  tools/precompute_qwen_normavg_text_cache.py         Qwen normavg 缓存
  tools/llm2vec_qwen3/                                Qwen3 LLM2Vec 全套（含 4 个自检脚本）
  run_logs/launch/hml3d_ablation_scheduler_20260731.sh  GPU 调度器

权重
  /scratch/pf2m24/hf-models/Meta-Llama-3-8B-Instruct
  /scratch/pf2m24/hf-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp{,-supervised}
  /scratch/pf2m24/hf-models/Qwen3-8B{,-mntp,-mntp-supervised}

缓存（均为正确 chat 格式）
  /scratch/pf2m24/text-caches/humanml3d_llm2vec_llama3_sup_chat_v2
  /scratch/pf2m24/text-caches/humanml3d_llm2vec_llama3_mntponly_chat_v2
  /scratch/pf2m24/text-caches/humanml3d_qwen3_l2v_sup_l128_v1
  /scratch/pf2m24/text-caches/kit_llm2vec_llama3_sup_chat_v2
  /scratch/pf2m24/text-caches/kit_qwen3_l2v_sup_l128_v1
  /scratch/pf2m24/text-caches/humanml3d_qwen3_8b_normavg_userchat_l128_v1
  /scratch/pf2m24/text-caches/echo-data                E5 有监督训练数据

外部参考
  /scratch/pf2m24/projects/Umdd/external_repos/llm2vec
  /scratch/pf2m24/projects/Umdd/external_repos/moge_UMO_ST
```
