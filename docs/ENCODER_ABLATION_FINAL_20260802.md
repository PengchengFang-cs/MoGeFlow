# 文本编码器消融：最终记录

**日期**：2026-08-02（实验于 08-01 03:39 启动，08-02 00:31 全部收尾）
**数据集**：HumanML3D（KIT 本轮全程暂停）
**状态**：六个 run 全部结束，权重已清理，`logs/` 记录完整保留

本文件取代 `TEXT_CONDITIONING_EXPERIMENTS_20260731.md` 的第 5 节结论——那一节的三条结论建立在被污染的 Llama 特征上，已全部撤回（详见第 6 节）。

---

## 1. 一句话结论

**三个编码器在 Top3 上无法区分（0.8711–0.8722，差 0.0011，远小于 0.0057 的评测噪声）；
FID 上 Qwen3-LLM2Vec 最终最低（0.0459）但需要最多 epoch；不带 pool 在两个选点上都更好，
带 pool 收敛更快且给出更均衡的检查点。**

---

## 2. 实验设计

六个 run＝三个文本编码器 × {不带 pooled，带 pooled}。每个 run **只有一个文本编码器**——
这是本轮的硬约束，排除了此前的 `dual_clip`（B/32+L/14）和 `clip_qwen`（CLIP-L 的 pooled +
Qwen3 的 tokens）两种双编码器方案。

| # | 文本编码器 | pooled 进 AdaLN | provider |
|---|---|---|---|
| 1 | CLIP-L/14 | 否 | `clip_large_tokenonly`（本轮新增） |
| 2 | CLIP-L/14 | 是 | `clip_large` |
| 3 | LLM2Vec-Llama3-supervised | 否 | `llm2vec_cache` |
| 4 | LLM2Vec-Llama3-supervised | 是 | `llm2vec_pooled_cache` |
| 5 | LLM2Vec-Qwen3-supervised | 否 | `llm2vec_cache` |
| 6 | LLM2Vec-Qwen3-supervised | 是 | `llm2vec_pooled_cache` |

「带 pool」与「不带 pool」的唯一差别是 `text_pooled_proj`（Linear 768→1152 + Linear
1152→1152），实测参数量差 **2,214,144**，token 流逐位相同。

**其余全部固定**：MoGeFlow-L（part_hidden_dim 192 / hidden_size 1152 / depth_double 6 /
depth_single 12 / 12 heads / mlp_ratio 4.0 / dropout 0.05，主干 690M，加 refiner 后可训练
733M）；refiner depth **2**（HY-Motion SingleTokenRefiner 的参考值，非变量），
`--text_refiner_pool content_span`；batch 64；lr 1e-4 half_cosine，warmup 2000，wd 0.01，
grad_clip 1.0，bf16；600 epoch 上限，seed 42，cond_drop 0.1，无 self-conditioning；
flow-only 损失（terminal/clean weight 均为 0），`tied_logits` + `codebook_nn`；
tokenizer 为 HF `new_vq_overlap_top3_20260529_best_top3.pth`。

**评测**：每 25 epoch 在 **test** 上一次，steps 96 / CFG 6.0 / repeat 1 / seed 42，
**continuous 解码**（代码中焊死，无参数入口）。

启动脚本：`run_logs/launch/hml3d_encoder6_worker_20260801.sh`

---

## 3. 定稿指标

每一行是一个检查点，行内所有数字都出自那一个检查点，**不跨检查点拼接**。

| # | 编码器 / pool | 停在 | 选点 | ep | FID | Top1 | Top2 | Top3 | MMDist | Div |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CLIP-L / 单个 | 500（早停） | best-FID | 400 | 0.0577 | .5804 | .7720 | .8573 | 2.6395 | 9.876 |
| 1 | | | best-Top3 | 175 | 0.1787 | .5873 | .7853 | **.8722** | 2.6242 | 10.310 |
| 2 | CLIP-L / 带pool | 425（早停） | best-FID | 325 | 0.0638 | .5748 | .7728 | .8562 | 2.6221 | 10.059 |
| 2 | | | best-Top3 | 200 | 0.1252 | **.5989** | .7834 | .8703 | **2.5944** | 10.290 |
| 3 | Llama3 / 单个 | 459（手动） | best-FID | 450 | 0.0639 | .5782 | .7746 | .8526 | 2.6497 | 9.960 |
| 3 | | | best-Top3 | 275 | 0.1233 | .5817 | .7819 | .8718 | 2.6160 | 10.103 |
| 4 | Llama3 / 带pool | 429（手动） | best-FID | 425 | 0.0689 | .5797 | .7787 | .8657 | 2.6058 | 9.965 |
| 4 | | | best-Top3 | 150 | 0.2140 | .5806 | .7772 | .8694 | 2.6302 | 10.303 |
| 5 | Qwen3 / 单个 | 598（手动） | best-FID | 525 | **0.0459** | .5696 | .7599 | .8526 | 2.6814 | 9.915 |
| 5 | | | best-Top3 | 175 | 0.2001 | .5793 | .7812 | .8711 | 2.6291 | 10.354 |
| 6 | Qwen3 / 带pool | 525（手动） | best-FID | 500 | 0.0501 | .5832 | .7718 | .8601 | 2.6122 | 9.988 |
| 6 | | | best-Top3 | 375 | 0.0593 | .5845 | .7797 | .8670 | 2.6049 | 10.090 |

---

## 4. 三个问题的答案

### 4.1 哪个编码器更好？——Top3 上分不出来，FID 上 Qwen3 最终最低但最慢

**Top3 的三个最好值：CLIP-L .8722 / Llama3 .8718 / Qwen3 .8711**，极差 0.0011。
单次评测的 Top3 标准差是 **0.0057**（见 5.2），所以**这三个数在统计上不可区分**。
论文不应声称任何一个编码器在检索对齐上更好。

FID 则有真实差异，但必须看轨迹而非终值。token-only 一侧的同轮次 FID：

| ep | CLIP-L | Llama3 | Qwen3 | 领先 |
|---|---|---|---|---|
| 50–150 | 1.1452→0.2249 | 0.6175→0.2093 | 0.5437→0.2162 | 两个 LLM2Vec 交替领先 |
| 175 | **0.1787** | 0.1827 | 0.2001 | CLIP-L |
| 250 | **0.1100** | 0.1382 | 0.1460 | CLIP-L |
| 325 | **0.0793** | 0.0966 | 0.0997 | CLIP-L |
| 400 | **0.0577** | 0.0737 | 0.0600 | CLIP-L |
| 425 | 0.0680 | 0.0724 | **0.0537** | Qwen3 |
| 450 | 0.0685 | 0.0639 | **0.0494** | Qwen3 |

**ep175–400 共 10 个连续轮次 CLIP-L 领先**，ep425 之后被 Qwen3 反超。原因是 CLIP-L
在 ep400 触底后开始回升（0.0577→0.0680→0.0685），而 Qwen3 一路降到 ep525。

所以准确表述是：**CLIP-L 收敛最快但最早触底；Qwen3 收敛最慢但底最深（0.0459，比
CLIP-L 低 20%）；Llama3 全程居中，最终 0.0639。** 说「Qwen3 的 LLM2Vec 更好」需要
限定为「在允许训练到 ep500+ 的前提下，FID 更低」。

### 4.2 带 pool 好还是不带 pool 好？——两者各赢一半，取决于你要什么

**按最好检查点，不带 pool 在三个编码器上全胜**：

| 编码器 | best-FID（单个 vs 带pool） | best-Top3（单个 vs 带pool） |
|---|---|---|
| CLIP-L | **0.0577** vs 0.0638 | **.8722** vs .8703 |
| Llama3 | **0.0639** vs 0.0689 | **.8718** vs .8694 |
| Qwen3 | **0.0459** vs 0.0501 | **.8711** vs .8670 |

**按同轮次效率，带 pool 在三个编码器上不输**（ep150 之后）：

| 编码器 | 重叠轮次 | FID 胜负（单个:带pool） | FID 平均差 | Top3 平均差 |
|---|---|---|---|---|
| CLIP-L | 12 | 3 : **9** | 带pool 低 0.0147 | 带pool 高 0.0010 |
| Llama3 | 12 | 6 : 6 | 带pool 低 0.0008 | 单个 高 0.0007 |
| Qwen3 | 16 | 5 : **11** | 带pool 低 0.0205 | 带pool 高 0.0025 |

**这两张表不矛盾，它们描述同一个现象**：带 pool 收敛更快，在中段大部分轮次上 FID 更低；
不带 pool 收敛更慢但最终触及更低的谷底。CLIP-L 和 Qwen3 上这个模式很清楚，Llama3 上
两者差距全部落在噪声内（FID 平均差 0.0008，Top3 0.0007，而噪声 σ 分别是 0.0064 和 0.0057）。

### 4.3 有没有 FID 和 Top3 都好的「均衡检查点」？——只有 run 6 接近

查遍全部评测记录，**没有任何一个检查点同时满足 FID ≤ 0.06 且 Top3 ≥ 0.87**。

最接近的是 **run 6（Qwen3 带pool）ep375：FID 0.0593 / Top3 0.8670**，Top3 差 0.003。

run 6 也是**唯一一个两个选点的 FID 都在 0.06 以内**的 run（best-FID ep500 是 0.0501，
best-Top3 ep375 是 0.0593）。其余五个的 best-Top3 检查点 FID 都在 0.12–0.21，差一个数量级——
它们的「高 Top3」只出现在训练早期，那时 FID 还很差。

**这是带 pool 最有价值的性质**：它把两个选点拉近了。若论文需要一行既真实又准确的数字，
run 6 是六个里唯一不用在 FID 和 Top3 之间做剧烈取舍的。

历史记录里有三个点满足该门槛，但都不可用：六月主线 ep290（0.0582/.8731）是 **nearest 解码**，
不是现行口径；`pscf_clipqwen3` 的 ep360/ep380（0.0588/.8707、.8716）是**双编码器**，违反本轮原则。

---

## 5. 方法学：写论文前必须知道的三件事

### 5.1 表里全部是 repeat-1，入表前必须补 repeat-20

第 3 节所有数字都是 `repeat_times=1`、`seed 42`。唯一做过 repeat-20 的检查点是六月主线的
`best_top3.pt`：

| 协议 | FID | Top3 |
|---|---|---|
| repeat-1（seed 42） | 0.0479 | 0.8739 |
| **repeat-20** | 0.0577 ± 0.0027 | **0.8608 ± 0.0025** |

**Top3 缩水 0.013，FID 涨 0.010。** 原因是结构性的：best-Top3 检查点是在 repeat-1 采样上
取最大值选出来的，必然吃到正向噪声。**第 3 节的所有 Top3 峰值都带这个偏差。**

### 5.2 噪声地板：Top3 σ = 0.0057，FID σ = 0.0064

同一个检查点、只换评测种子（42–61），20 次重复的实测分布：

| | 均值 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|
| FID | 0.0577 | **0.0064** | 0.0479 | 0.0718 |
| Top3 | 0.8608 | **0.0057** | 0.8513 | 0.8759 |

**任何小于这个量级的差异都不能作为结论。** 另外注意：`0.8608 ± 0.0025` 里的 ±0.0025 是
**均值的 95% CI**，不是单次评测的波动范围；单次评测的 ±2σ 是 **0.8493 ~ 0.8723**。

一个未解释的现象：20 次里有 2 次（seed 42、45）落在 0.874/0.876，与其余 18 次（0.851–0.864）
之间有 **0.0097 的断层**，而其余相邻档位间距都在 0.003 以内。这不是 i.i.d. 噪声的形状，
成因未查明。若要在论文里给区间，建议用 ±2σ 而非 CI，并考虑扩大重复次数。

### 5.3 检查点选择在 test 上

`models/codeflow/trainer.py:39` 的 `FULL_EVAL_SPLIT = "test"`——每 25 epoch 在 test 上评测，
并用 test 指标挑检查点。val 划分存在（HumanML3D 1460 条）但未使用。这是用户明确的决定，
记录在此以便撰稿时统一口径。

---

## 6. 本轮作废的实验及原因

`tools/precompute_llm2vec_text_cache.py` 有两个 bug，**只影响 LLM2Vec-Llama3 一条线**：

1. 上游 `!@#$%^&*()` 分隔符是位置标记（`tokenize` 用 `"".join(parts)` 剥掉），我们把它
   当文本喂进了 tokenizer。Llama-3 上切成 7 个 token，短 caption 里占 **37%**。
2. `pooled` 对全部 token 取均值，而上游用 `embed_mask` 只平均 caption 跨度。

**代价**：作废 5 个训练（07-31 的 ④'⑤'⑦'⑧ + KIT-Llama），以及三条结论——
「全局分支有益」「refiner 有害」「Qwen3 底座优于 Llama3」，全部撤回。这三条都是拿被污染的
Llama 运行作对照得出的。

**Qwen3 一侧从未受影响**，已用不依赖本仓库编码器代码的审计脚本逐字节证实：
200/200 条序列与 `PRE+caption+SUF` 吻合、与含分隔符的版本 0/200 吻合；`pooled` 与内容跨度
均值的相对误差 1.7e-3（bf16 舍入极限），与全 token 均值差 6.9e-2。

**修复后的验证**：新建的 Llama 缓存与上游 `llm2vec.LLM2Vec.encode()` 端到端比对，
16 条 caption **全部 cos = 1.000000**（`OFFICIAL_MATCH_OK`）。

**顺带发现**：这道闸门第一次运行时报 cos 0.15「缓存错误」，实为**闸门自己的参照侧写错**——
`LLM2Vec.from_pretrained` 缺 `merge_peft=True`，导致 MNTP 适配器被就地重初始化、supervised
权重键被 peft 静默丢弃，参照模型退化成与裸 Llama-3 逐位相同（实测 max|Δ| = 0.000000）。
现已修复并给参照侧加了与构建器同样的适配器自检。

---

## 7. 本轮的永久性代码变更

| 变更 | 位置 | 理由 |
|---|---|---|
| **`decode_mode` 焊死为 `continuous`** | `motion_code_flow.py` 的 `DECODE_MODE` 常量；`--decode_mode` 参数**从训练/评测/生成三处全部删除**；21 个 launch 脚本清理 | 曾有 4 个隐藏出口，其中 `gen_codeflow_t2m.py` 默认值和 `eval_t2m_cli.py` 的回退值都是 `nearest`，6 个 worker 的 KIT 分支写死 `nearest`。旧检查点的 options 里存着 `"nearest"`，恢复时会爬回来——已在 eval/gen 的恢复之后强制覆盖 |
| **`clip_large_tokenonly`** provider | `text_encoder.py` 的 `HFCLIPTextEncoder(use_pooled=)` | CLIP-L 此前无法关闭 pooled 分支；与 LLM2Vec provider 统一为 `output_dim=None` 开关 |
| **refiner 内容跨度池化** | `TextCondition.content_mask`、`--text_refiner_pool` | chat 模板脚手架占短 caption 约 26%，均值算子学不会忽略；只收窄 pooled 均值，注意力仍看全部有效 token |
| **闸门自检** | `verify_llm2vec_against_official.py` | 参照模型若与 base 逐位相同则直接报错退出 |
| **几何闸门** | `tools/llm2vec_qwen3/gate_encoder_geometry.py`（新增） | 12 组同义句对，去公共方向后要求严格检索 12/12 且 margin ≥ 0.40；Qwen3-v2 实测 margin +0.6848，24/24 |

### 早停规则（本轮引入并验证）

> FID 连续 **100** epoch 未刷新 **且** Top3 连续 **175** epoch 未刷新 → 停。

阈值来自实测：本仓库 21 个跑满 600 的历史运行中，再创新高前的最长平台期是 FID **80** epoch、
Top3 **170** epoch；best-FID 从未晚于 ep460，best-Top3 从未晚于 ep400。把规则回放到这 21 个
运行：**丢失 best 检查点 0 个**，平均省 118 epoch（20%），最多省 250。

安全性是结构性的：`best_fid.pt` / `best_top3.pt` 在达到的那一刻就写盘，停止不可能丢弃已有检查点。

脚本：`run_logs/launch/e6_early_stop_20260801.sh`

---

## 8. 现存资产

### 保留的检查点（110 GiB）

| run | 文件 | best-FID | best-Top3 | 用途 |
|---|---|---|---|---|
| `codeflow_part_structured_newvqtop3_w150_..._20260601` | `best_top3.pt` | 0.0568 | 0.8731 | 论文主表 HumanML3D 行 |
| `codeflow_pscf_hml3d_qwennormavg_w150_...` | 2 个 | **0.0407** | 0.8797 | T2 capacity **L**（690M） |
| `codeflow_pscf_hml3d_qwennormavg_w100_...` | 2 个 | 0.0538 | 0.8789 | T2 capacity **B**（307M） |
| `codeflow_pscf_hml3d_qwennormavg_w075_...` | 2 个 | 0.0772 | 0.8720 | T2 capacity **S**（173M） |
| `kit_l2vQ3sup_tok_r2_L_20260731` | 3 个 | **0.2248** | **0.8423** | KIT 最好（nearest 口径） |
| `codeflow_part_structured_kit_ddvq50k_bestfid_...rose09_g0` | 2 个 | 0.2697 | 0.8395 | KIT 次好 |

评测基础设施完整保留：`text_mot_match`、`length_estimator`、`rvq_nq6_dc512_*`、
`t2m_nlayer8_*`、`tres_*`。

### 已清理（567.8 GiB，20 个 run）

本轮 6 个编码器实验、07-31 的 3 个（abl9/abl10/r2）、`clipL_single`、`pscf_clipqwen3` ×2、
generic RVQ / code1024 ×5、encoder-latent、KIT 其余 2 个。
**权重不可恢复，但 `logs/full_eval.jsonl`、`logs/best_metrics.json`、`options.json` 全部保留**——
所有指标、逐轮次曲线、配置都可查。

### 文本缓存（全部保留，可直接重训）

```
/scratch/pf2m24/text-caches/humanml3d_llm2vec_llama3_sup_20260801     9.2 GiB  ← 通过官方比对
/scratch/pf2m24/text-caches/humanml3d_qwen3_l2v_sup_v2_20260801                ← Qwen3-v2（对齐配方）
/scratch/pf2m24/text-caches/humanml3d_qwen3_l2v_sup_l128_v1                    ← Qwen3-v1
/scratch/pf2m24/text-caches/kit_qwen3_l2v_sup_l128_v1
/scratch/pf2m24/text-caches/humanml3d_qwen3_8b_normavg_userchat_l128_v1
/scratch/pf2m24/text-caches/humanml3d_clip_l_qwen3_8b_hymotion_l128_v1
```

编码器权重：`/scratch/pf2m24/hf-models/Qwen3-8B-mntp-v2-20260801`、
`Qwen3-8B-mntp-supervised-v2-20260801`（对齐配方：80/10/10 掩码、warmup 0、loss_scale 50；
负样本 64，为官方 512 的 1/8，单卡无 GradCache）。

---

## 9. 写论文前还缺什么

1. **repeat-20 正式评测**。入表的每一个检查点都要补，否则 Top3 会缩水约 0.013（见 5.1）。
   本轮 6 个 run 的权重已删——**若要把它们写进论文，需要按保留的缓存重训**。
   目前保留权重的 6 个 run 可以直接补测。
2. **KIT 全部数字需要在 continuous 口径下重评**。现存 KIT 检查点的指标都是 nearest 时期的；
   decode 已焊死为 continuous，两者不可混用。权重还在，只需跑评测。
3. **T1 主表的 ± 值**。现行 caption 声称是 repeat-20 的 95% CI，但仓库里当时并无 repeat-20
   评测；07-30 已补跑，诚实值是 FID 0.0577±0.0027 / Top3 0.8608±0.0025（相对 SALAD 的 0.857
   优势从 1.7 个点缩到 0.4 个点，约一个 CI 半宽）。
4. **贡献 3（直接解码优于终端投影）的对照**。数字已在手（repeat-20：continuous
   0.0577±.0027 vs nearest 0.0686±.0027，区间不重叠），但 nearest 代码路径已移除——
   若需重现，要写独立脚本直接调 `terminal_ids()` + `tokenizer.decode_ids()`。

---

## 附：关键路径

```
本轮 6 个 run 的记录   checkpoints/t2m/hml3d_20260801_*/logs/
启动脚本               run_logs/launch/hml3d_encoder6_worker_20260801.sh
早停 / 续训            run_logs/launch/e6_early_stop_20260801.sh
                       run_logs/launch/e6_resume_all_20260801.sh
Llama 缓存前置链       run_logs/launch/e6_prep_llama_20260801.sh   （含官方比对闸门）
Qwen3 重训前置链       run_logs/launch/e6_prep_qwen3_20260801.sh   （含几何闸门）
上游参考实现           /scratch/pf2m24/projects/Umdd/external_repos/llm2vec
论文                   /iridisfs/scratch/pf2m24/projects/Umdd/overleaf-paper/iclr2026_conference.tex
```
