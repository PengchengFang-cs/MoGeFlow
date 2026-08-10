# Graph-RVQ + Graph-CodeFlow 管线说明

本文档解释当前代码里“Graph-VQVAE tokenizer + Graph-CodeFlow backbone”的核心信息流，重点回答：

1. RVQ 是怎么量化的。
2. backbone 训练时学的是什么。
3. 推理时连续预测怎么一步步 nearest/snap 成 RVQ indices。
4. 最后怎么用冻结 tokenizer decode 回动作。

这条线和前面的 Gaussian VAE / latent diffusion 是分开的。它复用 graph attention / temporal attention / AnyTop dataset 这些模块，但模型和训练脚本在 `vq_model/` 与 `CodeFlow_Model/` 下独立维护。

---

## 0. 符号

常用 shape：

- `B`: batch size
- `T`: fine motion 帧数，当前全长训练一般是 `300`
- `T_lat`: latent 时间长度，通常是 `ceil(T / temporal_stride)`；stride=4 时 `300 -> 75`。如果 clip 长度不能整除 stride，导出/推理会多生成到下一个 latent frame，再在渲染前裁回原始 `T`。
- `J`: fine joints，任意拓扑，每个样本不同，batch 内 pad 到 `max_joints`
- `C`: coarse slots，EdgeSegmentPool 后的有效 slot 数，batch 内 pad 到 `max_coarse`
- `D`: latent/channel dim，当前通常 `512`
- `Q`: RVQ residual stages，当前通常 `4`
- `K`: 每个 codebook 的 code 数，当前通常 `512`
- `L`: caption token 长度，T5 token cache 通常 `64`

关键 tensor：

```text
motion              [B, J, 13, T]       AnyTopDataset 输出
motion_in           [B, T, J, 13]       送进 tokenizer encoder
h_lat               [B, T_lat, C, D]    RVQ 之前的连续 coarse-slot latent
z_q                 [B, T_lat, C, D]    RVQ 之后的 summed code embedding
indices             [B, T_lat, C, Q]    每个 token 的 Q 个 code id
token_mask          [B, T_lat, C]       有效 latent token
pooled_adjacency    [B, C, C]           coarse graph 邻接
pooled_geodesic     [B, C, C]           coarse graph hop distance
caption_emb         [B, 768]            T5 mean pooled 文本
caption_token_emb   [B, L, 768]         T5 token-level 文本
```

---

## 1. Graph-VQVAE Tokenizer 总体流程

核心文件：

- `src/models/vq_model/graph_vq_tokenizer.py`
- `src/models/vq_model/quantizer.py`
- `src/models/vq_model/masked_motion_decoder.py`
- `scripts/train_graph_vqvae.py`
- `src/models/vq_model/losses.py`

`GraphVQTokenizer` 顶部 docstring 已经把主流程写得很清楚：

代码位置：

- `src/models/vq_model/graph_vq_tokenizer.py:1-30`

实际信息流：

```text
AnyTop 13ch motion
  [B, J, 13, T]
    -> permute
  [B, T, J, 13]
    -> SkeletonEncoder + SlotNorm
  [B, T, J, D]
    -> EdgeSegmentPool
  h_lat [B, T_lat, C, D]
    -> optional pre-VQ graph-temporal layers
  h_lat [B, T_lat, C, D]
    -> MaskedResidualVQ
  z_q [B, T_lat, C, D], indices [B, T_lat, C, Q]
    -> optional post-VQ graph-temporal layers
    -> temporal repeat_interleave
  [B, T, C, D]
    -> MaskedMotionDecoder slot-to-joint
  feats [B, T, J, D]
    -> anytop13 head
  pred_motion [B, T, J, 13]
```

核心代码：

- tokenizer 构造 encoder/pool/RVQ/decoder：`src/models/vq_model/graph_vq_tokenizer.py:98-184`
- encoder + EdgeSegmentPool：`src/models/vq_model/graph_vq_tokenizer.py:186-257`
- decode：`src/models/vq_model/graph_vq_tokenizer.py:259-293`
- full forward：`src/models/vq_model/graph_vq_tokenizer.py:486-513`

人话解释：

Tokenizer 的作用是把任意拓扑骨架动作先变成规则的 coarse-slot latent grid，再用 RVQ 变成离散 code。它不是在 fine joint 上直接做 VQ，而是在我们图池化后的 `T_lat x C` token map 上做 VQ。

---

## 2. RVQ 之前：图感知 coarse latent 怎么来

入口在：

- `GraphVQTokenizer.encode()`：`src/models/vq_model/graph_vq_tokenizer.py:186-257`

关键步骤：

1. Dataset 给出 `batch.anytop_x [B,J,13,T]`。
2. 先转成 `[B,T,J,13]`：
   - `src/models/vq_model/graph_vq_tokenizer.py:199`
3. `SkeletonEncoder` 同时编码 motion 和 skeleton：
   - motion path 得到 `h0 [B,T,J,D]`
   - skeleton-only path 得到 `s_j [B,J,D]`
   - `src/models/vq_model/graph_vq_tokenizer.py:202-213`
4. `SlotNorm` 后进入 `EdgeSegmentPool`：
   - `src/models/vq_model/graph_vq_tokenizer.py:214-225`
5. Pool 输出：
   - `h_lat / pooled_features [B,T_lat,C,D]`
   - `assignment [B,J,C]`
   - `coarse_mask [B,C]`
   - `frame_mask_lat [B,T_lat]`
   - `pooled_adjacency [B,C,C]`
   - `pooled_geodesic [B,C,C]`
   - `pooled_skeleton_embeddings [B,C,D]`
   - `src/models/vq_model/graph_vq_tokenizer.py:226-236`

然后可选的 pre-VQ graph-temporal refine：

- `src/models/vq_model/graph_vq_tokenizer.py:241-243`

这里的 `CoarseGraphTemporalLayer` 是：

- spatial: 对每个 latent frame，在 `C` 个 coarse slots 上做 graph attention。
- temporal: 对每个 coarse slot，在 `T_lat` 上做 temporal attention。

代码位置：

- `src/models/vq_model/graph_vq_tokenizer.py:49-95`

---

## 3. RVQ 量化：一个 token 变成 Q 个 code id

核心类：

- `MaskedResidualVQ`
- `src/models/vq_model/quantizer.py:231-420`

每个有效 token 是一个 `D` 维向量：

```text
x = h_lat[b, t, c]   [D]
```

RVQ 不是只查一次 codebook，而是 residual quantization：

```text
residual_0 = x

for q in 0..Q-1:
    id_q = nearest_codebook_q(residual_q)
    e_q  = codebook_q[id_q]
    residual_{q+1} = residual_q - e_q

z_q_token = e_0 + e_1 + ... + e_{Q-1}
indices_token = [id_0, id_1, ..., id_{Q-1}]
```

代码对应：

- nearest lookup `quantize()`：`src/models/vq_model/quantizer.py:164-178`
- residual loop：`src/models/vq_model/quantizer.py:342-367`
- accumulated `q_total`：`src/models/vq_model/quantizer.py:307-308,365`
- output `indices [B,T_lat,C,Q]`：`src/models/vq_model/quantizer.py:345-349`
- straight-through estimator：`src/models/vq_model/quantizer.py:411-417`

mask 规则很重要：

- `token_mask=False` 的 padded token 不参与 codebook EMA。
- padded token 的 `indices=-1`。
- padded token 的 `z_q=0`。

代码位置：

- mask-aware contract：`src/models/vq_model/quantizer.py:1-35`
- padded indices 置 `-1`：`src/models/vq_model/quantizer.py:345-349`
- padded token zero + zero grad：`src/models/vq_model/quantizer.py:411-417`

---

## 4. Graph-VQVAE Tokenizer 训练目标

训练脚本：

- `scripts/train_graph_vqvae.py`

模型构造参数：

- `scripts/train_graph_vqvae.py:276-366`
- `scripts/train_graph_vqvae.py:476-495`

训练 step：

- forward：`scripts/train_graph_vqvae.py:619-627`
- gate 检查 `z_q / indices` shape：`scripts/train_graph_vqvae.py:628-637`
- loss：`scripts/train_graph_vqvae.py:639-641`
- backward/step：`scripts/train_graph_vqvae.py:650-685`

VQ loss 文件：

- `src/models/vq_model/losses.py`

总 loss：

```text
L_total =
    w_pos     * pos
  + w_rot     * rot
  + w_vel     * vel
  + w_contact * contact
  + w_world   * world
  + w_fk      * fk
  + w_traj    * traj
  + w_commit  * commit
```

代码位置：

- loss 公式注释：`src/models/vq_model/losses.py:1-21`
- pos/rot/vel/contact：`src/models/vq_model/losses.py:103-111`
- world/fk/traj：`src/models/vq_model/losses.py:113-123`
- commit loss：`src/models/vq_model/losses.py:125-148`
- total assembly：`src/models/vq_model/losses.py:149-155`

注意：

- 这里没有 KL，因为 VQ 替代了 Gaussian latent。
- `commit` 是 RVQ 的 commitment loss，用来让 encoder output 靠近选中的 code。
- `w_commit` 只在 loss wrapper 里乘一次。

---

## 5. Token Export：为什么 backbone 不在线跑 tokenizer encoder

训练 CodeFlow backbone 前，先用冻结的 Graph-VQVAE tokenizer 离线导出 token cache。

注意区分两个阶段：

- 训练 Graph-VQVAE tokenizer 时，encoder、RVQ codebook、decoder/head 都在训练。
- 训练 CodeFlow backbone 和推理时，Graph-VQVAE tokenizer 才冻结，只负责提供 `z_q / indices / graph meta` 或把预测 token decode 回动作。

脚本：

- `scripts/export_graph_vq_tokens.py`

它对每条 motion 保存：

```text
z_q                         [T_lat,C,D]
indices                     [T_lat,C,Q]
token_mask                  [T_lat,C]
coarse_mask                 [C]
frame_mask_lat              [T_lat]
pooled_adjacency            [C,C]
pooled_geodesic             [C,C]
pooled_skeleton_embeddings  [C,D]
assignment                  [J,C]
s_j                         [J,D]
joint_mask                  [J]
caption_emb                 [768]
caption_token_emb           [L,768]
caption_token_mask          [L]
```

代码位置：

- export schema 注释：`scripts/export_graph_vq_tokens.py:1-31`
- load frozen tokenizer：`scripts/export_graph_vq_tokens.py:57-82`
- dataset + caption cache 打开：`scripts/export_graph_vq_tokens.py:164-176`
- tokenizer encode + quantizer：`scripts/export_graph_vq_tokens.py:250-258`
- 保存 `z_q / indices / graph meta / text`：`scripts/export_graph_vq_tokens.py:260-319`
- RVQ identity audit：`scripts/export_graph_vq_tokens.py:285-332`

这里有一个关键自检：

```text
ids_to_embeddings(indices) ≈ z_q
```

也就是从导出的 indices 重新查 codebook，应该能还原导出的 `z_q`。这是保证后续 snap/decode 正确的核心 gate。

---

## 6. CodeFlow 训练数据怎么读

Dataset：

- `src/models/CodeFlow_Model/token_dataset.py`

`TokenCacheDataset` 不读原始 motion，它读 export 出来的 `.npz`：

- `z_q [T_lat,C,D]`
- `indices [T_lat,C,Q]`
- graph metadata
- text embedding
- decode metadata

代码位置：

- dataset schema：`src/models/CodeFlow_Model/token_dataset.py:1-14`
- `__getitem__` 读 `.npz`：`src/models/CodeFlow_Model/token_dataset.py:45-75`
- collate：`src/models/CodeFlow_Model/token_dataset.py:78-90`

所以 backbone 训练阶段的 target 是冻结 tokenizer 的 `z_q`，不是原始 motion，也不是 pre-RVQ `h_lat`。

---

## 7. CodeFlow Backbone 学什么

核心 wrapper：

- `src/models/CodeFlow_Model/flow.py`

正式 backbone：

- `src/models/CodeFlow_Model/graph_pscf.py`

旧 Level-A probe：

- `src/models/CodeFlow_Model/graph_codeflow.py`

当前正式配置默认走：

```text
GraphCodeFlow(model_variant="graph_pscf")
```

代码位置：

- variant selector：`src/models/CodeFlow_Model/flow.py:55-99`
- train script 默认 `graph_pscf`：`scripts/train_graph_codeflow.py:243-253`

训练目标是 rectified flow velocity，不是直接分类 code id：

```text
x = normalize(z_q)
noise ~ N(0, I)
t ~ Uniform(0, 1)

z_t = t * x + (1 - t) * noise
v_target = x - noise

v_pred = backbone(z_t, t, text, graph)
loss = MSE(v_pred, v_target) over valid token_mask
```

代码位置：

- flow math 注释：`src/models/CodeFlow_Model/flow.py:1-30`
- empirical norm：`src/models/CodeFlow_Model/flow.py:108-126`
- flow loss：`src/models/CodeFlow_Model/flow.py:170-236`
- train step：`scripts/train_graph_codeflow.py:547-574`

训练时的 batch cond：

- `caption_emb [B,768]`
- `caption_token_emb [B,L,768]`
- `caption_token_mask [B,L]`
- `has_text [B]`，训练时会按 `cond_drop_prob` 做 CFG dropout
- `pooled_adjacency / pooled_geodesic`
- `pooled_skeleton_embeddings`
- `coarse_mask / frame_mask_lat`

代码位置：

- `build_cond()`：`scripts/train_graph_codeflow.py:90-113`
- CFG text drop：`scripts/train_graph_codeflow.py:99-102`

---

## 8. Graph-PSCF Backbone 的结构

核心文件：

- `src/models/CodeFlow_Model/graph_pscf.py`
- `src/models/CodeFlow_Model/dit_blocks.py`

Graph-PSCF 有三条流：

```text
slot stream   h_slot  [B, T_lat, C, D]   RVQ latent grid，保留 coarse graph 结构
frame stream  h_frame [B, T_lat, D]      每个 latent frame 一个 holder token
text stream   h_text  [B, L, D]          T5 token stream
```

代码位置：

- 三流说明：`src/models/CodeFlow_Model/graph_pscf.py:12-31`
- forward 输入检查：`src/models/CodeFlow_Model/graph_pscf.py:336-421`

### 8.1 slot stream：图感知

slot stream 每层会做：

1. graph-spatial attention over `C`
2. temporal attention over `T_lat`
3. strict mask

代码位置：

- `GraphSlotTemporalBlock`：`src/models/CodeFlow_Model/graph_pscf.py:66-125`
- graph attention over coarse slots：`src/models/CodeFlow_Model/graph_pscf.py:99-113`
- temporal attention：`src/models/CodeFlow_Model/graph_pscf.py:115-125`

这说明 backbone 是图感知的：它直接使用 `pooled_adjacency` 和 `pooled_geodesic`。

### 8.2 frame holder：压缩时间级全局上下文

`h_frame [B,T_lat,D]` 不是替代 `h_slot [B,T_lat,C,D]`，而是每一帧一个全局 holder。

交互方式：

1. holder 读同一帧的所有 coarse slots。
2. holder 更新后再 broadcast/add 回该帧所有 slots。

代码位置：

- `GraphFrameSlotCoupling`：`src/models/CodeFlow_Model/graph_pscf.py:128-220`
- holder cross-attn 读 slots：`src/models/CodeFlow_Model/graph_pscf.py:184-206`
- holder 注回 slots：`src/models/CodeFlow_Model/graph_pscf.py:208-219`

### 8.3 text stream：对齐 CodeFlow/FLUX 式 double/single stream

两路文本：

1. `text_global [B,768] -> pooled_text -> cond`
   - 用于 timestep/AdaLN/FiLM 调制。
2. `text_tokens [B,L,768] -> h_text`
   - 和 frame stream 做 double/single stream attention。

代码位置：

- text pooled/token projection：`src/models/CodeFlow_Model/graph_pscf.py:279-282`
- `cond = timestep_emb + pooled_text`：`src/models/CodeFlow_Model/graph_pscf.py:425-430`
- token stream：`src/models/CodeFlow_Model/graph_pscf.py:444-447`

double/single DiT block 来自本地 port：

- `src/models/CodeFlow_Model/dit_blocks.py`
- double stream block：`src/models/CodeFlow_Model/dit_blocks.py:241-303`
- single stream block：`src/models/CodeFlow_Model/dit_blocks.py:306-341`

Graph-PSCF 主干：

- double stage：`src/models/CodeFlow_Model/graph_pscf.py:453-480`
- single stage：`src/models/CodeFlow_Model/graph_pscf.py:481-509`
- output velocity：`src/models/CodeFlow_Model/graph_pscf.py:511-513`

默认深度：

- `depth_double=6`
- `depth_single=12`

代码位置：

- `src/models/CodeFlow_Model/graph_pscf.py:238-250`
- `scripts/train_graph_codeflow.py:259-262`

---

## 9. Backbone 训练循环

训练脚本：

- `scripts/train_graph_codeflow.py`

流程：

```text
TokenCacheDataset
  -> batch:
      z_q, token_mask, graph meta, text
  -> build_cond()
  -> GraphCodeFlow.flow_loss()
  -> loss.backward()
  -> optimizer.step()
```

代码位置：

- load frozen tokenizer：`scripts/train_graph_codeflow.py:65-87`
- load token cache：`scripts/train_graph_codeflow.py:367-373`
- empirical z_q stats：`scripts/train_graph_codeflow.py:116-190`
- build model：`scripts/train_graph_codeflow.py:418-430`
- install empirical norm：`scripts/train_graph_codeflow.py:431-458`
- training dataloader：`scripts/train_graph_codeflow.py:489-500`
- train step：`scripts/train_graph_codeflow.py:547-574`
- projection QA：`scripts/train_graph_codeflow.py:193-240`
- validation + checkpoint：`scripts/train_graph_codeflow.py:607-638`

训练 loss 只有 flow MSE：

```text
loss = flow_loss_weight * flow_loss
```

代码位置：

- `scripts/train_graph_codeflow.py:552-555`

`terminal_loss_weight` 和 `clean_loss_weight` 参数存在，但当前代码没有把它们接入 loss；正式训练是 flow-only。

projection QA 的含义也要分清：训练日志里的 projection QA 不是完整 ODE 采样。它是在某个随机 `t` 上用当前模型反推一次 clean latent，再做 residual nearest snap，用来快速看 `z_hat -> z_snap` 的投影误差。完整生成质量仍然要看推理脚本里的 ODE sample + snap + decode。

---

## 10. 推理：从 skeleton + text 生成 motion

推理脚本：

- `scripts/animate_graph_codeflow.py`

脚本顶部 docstring 写了完整推理路径：

- `scripts/animate_graph_codeflow.py:1-17`

实际步骤：

### Step 1. 读目标 skeleton 和 prompt

Dataset 取一个样本，作为目标骨架；motion 本身不作为生成输入，只用来拿 skeleton / caption / 可选 GT QA。

代码位置：

- dataset：`scripts/animate_graph_codeflow.py:159-166`
- batch：`scripts/animate_graph_codeflow.py:180-187`

### Step 2. skeleton-only pool，得到 coarse graph meta

```text
tokenizer.prepare_skeleton_only(batch, T_lat_i)
```

输出：

```text
s_j
assignment [B,J,C]
coarse_mask [B,C]
frame_mask_lat [B,T_lat]
token_mask [B,T_lat,C]
pooled_adjacency [B,C,C]
pooled_geodesic [B,C,C]
pooled_skeleton_embeddings [B,C,D]
```

代码位置：

- prepare_skeleton_only 实现：`src/models/vq_model/graph_vq_tokenizer.py:414-465`
- animate 调用：`scripts/animate_graph_codeflow.py:204-216`

### Step 3. ODE + CFG 采样连续 latent `z_hat`

```text
z_hat = flow.sample(cond, token_mask, T_lat, C, steps, cfg_scale)
```

这里从高斯噪声开始，用 learned velocity field 走 ODE：

```text
z_0 ~ noise
for i in 0..steps-1:
    v_cond   = backbone(z_i, t_i, cond)
    v_uncond = backbone(z_i, t_i, cond with text dropped)
    v = v_uncond + cfg_scale * (v_cond - v_uncond)
    z_{i+1} = z_i + dt * v

z_hat = denormalize(z_final)
```

代码位置：

- sampler：`src/models/CodeFlow_Model/flow.py:241-291`
- animate 调用：`scripts/animate_graph_codeflow.py:217-221`

注意：此时 `z_hat [B,T_lat,C,D]` 是连续向量，还不是合法的 RVQ code sequence。

### Step 4. residual nearest snap：连续 `z_hat` 变回 RVQ indices

```text
proj = tokenizer.nearest_residual_ids(z_hat, token_mask)
indices_hat = proj["indices_hat"]   [B,T_lat,C,Q]
z_snap      = proj["z_snap"]         [B,T_lat,C,D]
```

算法和训练时 RVQ residual loop 一样：

```text
residual = z_hat
for q in 0..Q-1:
    id_q = nearest_codebook_q(residual)
    e_q  = codebook_q[id_q]
    residual = residual - e_q
z_snap = sum_q e_q
indices_hat = stack(id_q)
```

代码位置：

- nearest residual API：`src/models/vq_model/graph_vq_tokenizer.py:346-412`
- animate 调用：`scripts/animate_graph_codeflow.py:217-225`
- training QA 里也用它：`scripts/train_graph_codeflow.py:193-240`

这一步的 `projection_error` 是关键诊断：

```text
projection_error = MSE(z_hat, z_snap) over valid tokens
```

如果 continuous decode 很好、snapped decode 很差，说明问题发生在 continuous latent 到 codebook 的投影之后；是否真的是“已经靠近 codebook 流形但 snap 不好”，还要结合 `projection_error` 看。如果两者都差，则更像是 flow/backbone 本身没学好。

### Step 5. 从 indices decode 回 motion

```text
pred = tokenizer.decode_from_indices(indices_hat, meta, batch)
```

内部做两步：

1. `ids_to_embeddings(indices_hat)`：
   - 每个 stage 查 codebook embedding。
   - 把 Q 个 residual embedding 相加。
   - 得到 `z_q / z_snap [B,T_lat,C,D]`。
2. 调 `decode(z_q, meta, batch)`：
   - post-VQ refine
   - temporal upsample
   - slot-to-joint decoder
   - anytop13 head

代码位置：

- indices -> embeddings：`src/models/vq_model/graph_vq_tokenizer.py:306-344`
- decode_from_indices：`src/models/vq_model/graph_vq_tokenizer.py:467-484`
- decode 主体：`src/models/vq_model/graph_vq_tokenizer.py:259-293`
- animate 调用：`scripts/animate_graph_codeflow.py:222-226`

### Step 6. anytop13 motion -> world positions -> GIF

输出 `pred_motion [B,T,J,13]` 还是 normalized AnyTop 13ch。渲染前要反归一化，然后走 rot6d FK 恢复世界坐标：

```text
raw = pred_norm * (std + floor) + mean
world = recover_from_bvh_rot_np(raw, parents, offsets)
```

代码位置：

- de-normalize + FK recovery：`scripts/animate_graph_codeflow.py:228-241`
- GIF 渲染：`scripts/animate_graph_codeflow.py:256-270`

---

## 11. 一句话版信息流

### Tokenizer 训练

```text
motion [B,J,13,T]
  -> graph encoder
  -> EdgeSegmentPool
  -> h_lat [B,T_lat,C,D]
  -> RVQ
  -> z_q + indices
  -> trainable decoder/head during tokenizer training
  -> pred_motion [B,T,J,13]
  -> reconstruction + geometry + commit loss
```

### Backbone 训练

```text
frozen tokenizer export:
  motion -> z_q [T_lat,C,D] + indices [T_lat,C,Q] + graph/text metadata

CodeFlow training:
  z_q + noise + t -> z_t
  graph/text-conditioned backbone predicts v
  MSE(v_pred, z_q - noise)
```

### 推理

```text
skeleton + text
  -> skeleton-only EdgeSegmentPool meta
  -> ODE sample continuous z_hat [B,T_lat,C,D]
  -> nearest_residual_ids(z_hat)
  -> indices_hat [B,T_lat,C,Q]
  -> ids_to_embeddings(indices_hat) = z_snap
  -> frozen tokenizer decode
  -> pred_motion [B,T,J,13]
  -> FK/world recovery
  -> GIF
```

---

## 12. 给同事讲时的重点

1. 我们的 VQ token 不是人体固定 joint token，而是任意拓扑骨架经过 `EdgeSegmentPool` 后的 coarse-slot token。
2. RVQ 的离散 id 是 `[T_lat, C, Q]`，每个 `(t,c)` 有 `Q` 个 residual code。
3. Backbone 不直接预测 id，也不是 latent diffusion；它在 post-RVQ 的连续 `z_q` 空间做 rectified flow。
4. 推理时先生成连续 `z_hat`，再逐 stage residual nearest 到 codebook，得到完整 `indices_hat [T_lat,C,Q]`。
5. 最终动作不是 backbone 直接输出的，而是冻结 Graph-VQVAE decoder 从 snapped RVQ indices 解出来的。
6. 图感知来自两处：
   - tokenizer encoder/pool/decode 使用 skeleton graph。
   - Graph-PSCF backbone 的 slot stream 使用 `pooled_adjacency / pooled_geodesic` 做 graph attention。
7. 文本也是两路：
   - T5 mean pooled global text 进入 AdaLN/FiLM 条件。
   - T5 token sequence 进入 double/single stream attention。
