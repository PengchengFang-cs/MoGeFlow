import torch.nn as nn
import torch
from kvctrl.models.encdec import Encoder, Decoder
from kvctrl.models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset

import numpy as np
import json
# class VQVAE_limb_hml(nn.Module):
#     def __init__(self,
#                  args,
#                  nb_code=1024,
#                  code_dim=512,
#                  output_emb_width= 512,
#                  down_t=3,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
        
#         super().__init__()
#         self.code_dim = code_dim
#         self.num_code = nb_code
#         self.quant = args.quantizer
#         #self.partSeg = [[0], [3, 6, 9, 12, 15], [14, 17, 19, 21], [13, 16, 18, 20], [2, 5, 8, 11], [1, 4, 7, 10]]
#         def values_term(i):
#             i -= 1
#             return [4+i*3, 4+i*3+1, 4+i*3+2] + [4 + 63+i*6 + k for k in range(6)] + [4+63+126+(i+1)*3 + k for k in range(3)]
#         self.partSeg = [[0, 1, 2, 3, 4+63+126, 4+63+126+1, 4+63+126+2],
#                         [x for i in [3, 6, 9, 12, 15] for x in values_term(i)],
#                         [x for i in [13, 16, 18, 20] for x in values_term(i)],
#                         [x for i in [14, 17, 19, 21] for x in values_term(i)],
#                         [x for i in [1, 4, 7, 10] for x in values_term(i)] + [259, 260],
#                         [x for i in [2, 5, 8, 11] for x in values_term(i)] + [261, 262]]
#         #single_limb_emb_width = output_emb_width // len(self.partSeg)
#         #self.encoder = Encoder(22 * args.data_rot_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
#         self.decoder = Decoder(263, output_emb_width * len(self.partSeg), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
#         if args.quantizer == "ema_reset":
#             self.quantizers = nn.ModuleList([QuantizeEMAReset(nb_code, code_dim, args) for part in self.partSeg])
#         elif args.quantizer == "orig":
#             self.quantizers = nn.ModuleList([Quantizer(nb_code, code_dim, 1.0) for part in self.partSeg]) 
#         elif args.quantizer == "ema":
#             self.quantizer = QuantizeEMA(nb_code, code_dim, args)
#         elif args.quantizer == "reset":
#             self.quantizers = nn.ModuleList([QuantizeReset(nb_code, code_dim, args) for part in self.partSeg])
#         self.limb_encoders = nn.ModuleList([
#             Encoder(len(part), output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
#             for part in self.partSeg
#         ])
        
        
#     def preprocess(self, x):
#         # (bs, T, Jx3) -> (bs, Jx3, T)
#         x = x.permute(0,2,1).float()
#         return x

#     def postprocess(self, x):
#         # (bs, Jx3, T) ->  (bs, T, Jx3)
#         x = x.permute(0,2,1)
#         return x

#     def forward(self, x):
        
#         #x_in = self.preprocess(x)
#         # Encode
#         #x_s = [self.limb_encoders[i](x_in[:, part, :]) for i, part in enumerate(self.partSeg)]
        
#         # x = x.squeeze()
#         x = x.permute(0,2,1)
#         bs, F, T = x.shape
#         x_in = x.permute(0, 2, 1)
#         x_s = []
        
#         for i, part in enumerate(self.partSeg):
#             x_current = x_in[:, :, part].reshape(bs, T, -1).cuda()
#             x_current = self.preprocess(x_current)
#             x_feature = self.limb_encoders[i](x_current)
            
#             #x_feature.shape => (Batch, 128, 49)
#             #torch.norm(x_feature, dim=[1, 2]).shape => (Batch, 1, 1)
#             x_feature = x_feature / torch.norm(x_feature, dim=[1, 2]).unsqueeze(1).unsqueeze(1)
            
#             x_s.append(x_feature)
            
#         x_quantized = []
#         loss = .0
#         perplexity = .0
#         for i, part in enumerate(self.partSeg):
#             x, l, p  = self.quantizers[i](x_s[i])
#             x_quantized.append(x)
#             loss += l
#             perplexity += p
#         x_quantized = torch.cat(x_quantized, dim = 1)
        

#         ## decoder
#         x_decoder = self.decoder(x_quantized)
        
#         x_out = x_decoder.unsqueeze(2)
#         #x_out = self.postprocess(x_decoder)
#         return x_out, loss, perplexity
    
#     def get_quantized_codes(self, x, in_idx_format = True):
#         x = x.squeeze(-2)
#         bs, F, T = x.shape
#         x_in = x.permute(0, 2, 1)
#         x_s = []
#         for i, part in enumerate(self.partSeg):
#             x_current = x_in[:, :, part].reshape(bs, T, -1).cuda()
#             x_current = self.preprocess(x_current)
#             x_feature = self.limb_encoders[i](x_current)
            
#             x_feature = x_feature / torch.norm(x_feature, dim=[1, 2]).unsqueeze(1).unsqueeze(1) 
            
#             x_s.append(x_feature)
#         if not in_idx_format:
#             x_quantized = []
#             for i, part in enumerate(self.partSeg):
#                 x, l, p  = self.quantizers[i](x_s[i])
#                 x_quantized.append(x)
#             return x_quantized
#         else:
#             x_ids = []
#             for i, part in enumerate(self.partSeg):
#                 x_ids.append(self.quantizers[i].get_code_idx(x_s[i]).unsqueeze(-2))
#             x_ids = torch.cat(x_ids, dim = 1)
#             return x_ids
    
#     def forward_decoder(self, x_id):
#         # x.shape = ()
#         x_quantized = []
#         for i, part in enumerate(self.partSeg):
#             x_current = self.quantizers[i].forward_from_code_idx(x_id[:, :, i])
#             x_quantized.append(x_current)
#         x_quantized = torch.cat(x_quantized, dim = 1)

#         ## decoder
#         x_decoder = self.decoder(x_quantized)
#         x_out = self.postprocess(x_decoder)
#         return x_out
    
#     def get_x_quantized_from_x_ids(self, x_id):
#         x_quantized = []
#         #print(f"inside   {x_id.shape}")
#         for i, part in enumerate(self.partSeg):
#             x_current = self.quantizers[i].forward_from_code_idx(x_id[..., i])
#             x_quantized.append(x_current)
#         return x_quantized

#     def forward_decoder_from_quantized_codes(self, x_quantized):
#         x_quantized = torch.cat(x_quantized, dim = 1)
#         x_decoder = self.decoder(x_quantized)
#         x_out = x_decoder.unsqueeze(2)
#         return x_out

    
# class HumanVQVAE(nn.Module):
#     def __init__(self,
#                  args,
#                  nb_code=512,
#                  code_dim=512,
#                  output_emb_width=512,
#                  down_t=2,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
        
#         super().__init__()
        
        
#         self.nb_joints = 22
#         self.vqvae = VQVAE_limb_hml(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

#     def encode(self, x):
#         b, t, c = x.size()
#         quants = self.vqvae.encode(x) # (N, T)
#         return quants

#     def forward(self, x):
        
#         x_out, loss, perplexity = self.vqvae(x)
        
#         return x_out, loss, perplexity

#     def forward_decoder(self, x):
#         x_out = self.vqvae.forward_decoder(x)
#         return x_out
    
#     def get_code_idx(self, x):
#         return self.vqvae.get_quantized_codes(x)



def load_partition_from_file(path):
    """
    从 JSON 文件加载 partSeg，用于数据驱动的 skeleton 分块。
    文件格式: {"partSeg": [[dim_indices], ...], "n_parts": int, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["partSeg"]


class VQVAE_251(nn.Module):
    """
    多 body parts 版本（等价于之前的 VQVAE_limb_hml 思路）：
      - 输入: (B, T, input_dim)，HumanML3D=263, KIT=251
      - 6 个 limb encoder + 6 个 quantizer
      - 1 个全身 decoder
    """
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        self.dataname = args.dataname

        default_input_dim = 251 if self.dataname == "kit" else 263
        self.output_dim = int(getattr(args, "input_dim", default_input_dim))
        self.output_emb_width = output_emb_width

        # ======================
        #  body part segmentation (和你之前 VQVAE_limb_hml 一致)
        # 支持从 analyze_skeleton_partition.py 输出的 JSON 加载数据驱动分块
        # ======================
        partition_file = getattr(args, "partition_file", None)
        if partition_file is not None:
            self.partSeg = load_partition_from_file(partition_file)
            self.num_parts = len(self.partSeg)
        else:
            def values_term(i):
                i -= 1
                return [4 + i * 3, 4 + i * 3 + 1, 4 + i * 3 + 2] \
                       + [4 + 63 + i * 6 + k for k in range(6)] \
                       + [4 + 63 + 126 + (i + 1) * 3 + k for k in range(3)]

            self.partSeg = [
                [0, 1, 2, 3, 4+63+126, 4+63+126+1, 4+63+126+2],                        # root
                [x for i in [3, 6, 9, 12, 15] for x in values_term(i)],               # 脊椎
                [x for i in [13, 16, 18, 20] for x in values_term(i)],                # 左臂
                [x for i in [14, 17, 19, 21] for x in values_term(i)],                # 右臂
                [x for i in [1, 4, 7, 10] for x in values_term(i)] + [259, 260],      # 左腿
                [x for i in [2, 5, 8, 11] for x in values_term(i)] + [261, 262]       # 右腿
            ]
            self.num_parts = len(self.partSeg)  # = 6

        # ======================
        #  Encoders: 每个 part 一个
        # ======================
        self.limb_encoders = nn.ModuleList([
            Encoder(
                len(part), output_emb_width,
                down_t, stride_t, width, depth,
                dilation_growth_rate,
                activation=activation, norm=norm
            )
            for part in self.partSeg
        ])

        # ======================
        #  Decoder: 全身一个
        #   in_channels = output_emb_width * num_parts
        # ======================
        in_channels_decoder = output_emb_width * self.num_parts
        self.decoder = Decoder(
            self.output_dim,
            in_channels_decoder,
            down_t, stride_t, width, depth,
            dilation_growth_rate,
            activation=activation, norm=norm
        )

        # ======================
        #  Quantizers: 每个 part 一个
        # ======================
        def build_single_quantizer():
            if args.quantizer == "ema_reset":
                return QuantizeEMAReset(nb_code, code_dim, args)
            elif args.quantizer == "orig":
                return Quantizer(nb_code, code_dim, 1.0)
            elif args.quantizer == "ema":
                return QuantizeEMA(nb_code, code_dim, args)
            elif args.quantizer == "reset":
                return QuantizeReset(nb_code, code_dim, args)
            else:
                raise ValueError(f"Unknown quantizer type: {args.quantizer}")

        self.quantizers = nn.ModuleList(
            [build_single_quantizer() for _ in self.partSeg]
        )

    # ========== 形状变换 ==========
    def preprocess(self, x):
        """
        (B, T, C) -> (B, C, T)
        """
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        """
        (B, C, T) -> (B, T, C)
        """
        return x.permute(0, 2, 1)

    # ========== Encode：给上层 transformer 用的 code index ==========
    def encode(self, x):
        """
        x: (B, T, C)
        return: (B, T_latent * num_parts)
        （注意：这是 flatten 之后的版本，方便你原来 HumanVQVAE 'encode' 用）
        """
        B, T, C = x.shape
        device = x.device
        x_in = self.preprocess(x)        # (B, 263, T)
        x_in_T = x_in.permute(0, 2, 1)   # (B, T, 263)

        code_idx_parts = []
        T_latent_ref = None

        for i, part in enumerate(self.partSeg):
            # 取当前 limb: (B, T, len(part)) -> (B, len(part), T)
            x_part = x_in_T[:, :, part].to(device)
            x_part = x_part.permute(0, 2, 1)            # (B, len(part), T)

            # 编码
            z_e = self.limb_encoders[i](x_part)        # (B, D, T_latent)

            # 不做 norm，与训练版本一致
            # 量化（先 flatten）
            z_e_T = z_e.permute(0, 2, 1).contiguous()  # (B, T_latent, D)
            Bb, T_latent, D = z_e_T.shape
            if T_latent_ref is None:
                T_latent_ref = T_latent
            else:
                assert T_latent == T_latent_ref, "All limbs must have same T_latent"

            z_flat = z_e_T.view(-1, D)                 # (B*T_latent, D)
            idx_flat = self.quantizers[i].quantize(z_flat)   # (B*T_latent,)
            idx = idx_flat.view(Bb, T_latent)          # (B, T_latent)
            code_idx_parts.append(idx)

        # 拼在一起: (B, num_parts * T_latent)
        code_idx = torch.cat(code_idx_parts, dim=1)
        return code_idx

    # ========== get_quantized_codes：保持和之前 VQVAE_limb_hml 的接口 ==========
    def get_quantized_codes(self, x, in_idx_format=True):
        x = x.squeeze(-2)
        device = x.device
        self.to(device)  # 确保模型与输入在同一设备
        bs, F, T = x.shape
        x_in = x.permute(0, 2, 1)
        x_s = []
        for i, part in enumerate(self.partSeg):
            x_current = x_in[:, :, part].reshape(bs, T, -1).to(device)
            x_current = self.preprocess(x_current)
            x_feature = self.limb_encoders[i](x_current)

            x_s.append(x_feature)
        if not in_idx_format:
            x_quantized = []
            for i, part in enumerate(self.partSeg):
                x, l, p  = self.quantizers[i](x_s[i])
                x_quantized.append(x)
            return x_quantized
        else:
            x_ids = []
            for i, part in enumerate(self.partSeg):
                x_ids.append(self.quantizers[i].get_code_idx(x_s[i]).unsqueeze(-2))
            x_ids = torch.cat(x_ids, dim = 1)
            return x_ids

    # ========== Forward：训练 ==========
    def forward(self, x):
        """
        x: (B, T, C)
        return: x_out (B, T, C), loss, perplexity
        """
        x_in = self.preprocess(x)        # (B, 263, T)
        B, C, T = x_in.shape
        device = x_in.device

        x_in_T = x_in.permute(0, 2, 1)   # (B, T, 263)

        z_q_parts = []
        total_loss = 0.0
        total_perp = 0.0

        for i, part in enumerate(self.partSeg):
            x_part = x_in_T[:, :, part].to(device)     # (B, T, len(part))
            x_part = x_part.permute(0, 2, 1)          # (B, len(part), T)

            # 编码
            z_e = self.limb_encoders[i](x_part)       # (B, D, T_latent)

            # 不做 norm，与训练版本一致
            z_q, l, p = self.quantizers[i](z_e)       # (B, D, T_latent)

            z_q_parts.append(z_q)
            total_loss += l
            total_perp += p

        # 拼 channel: (B, D*num_parts, T_latent)
        z_q_all = torch.cat(z_q_parts, dim=1)

        # 全身 decoder
        x_decoder = self.decoder(z_q_all)             # (B, 263, T)
        x_out = self.postprocess(x_decoder)           # (B, T, 263)

        num_parts = float(self.num_parts)
        return x_out, total_loss / num_parts, total_perp / num_parts

    # ========== 从 code index 解码 ==========
    def forward_decoder(self, x_id):
        # x.shape = ()
        
        x_quantized = []
        for i, part in enumerate(self.partSeg):
            # import pdb; pdb.set_trace()
            x_current = self.quantizers[i].forward_from_code_idx(x_id[:, :, i]) # x_id[:, :, i]是 B 49 6
            x_quantized.append(x_current)
        x_quantized = torch.cat(x_quantized, dim = 1)
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out

    # ========== 只从 code index 得到 quantized feature list ==========
    def get_x_quantized_from_x_ids(self, x_id):
        x_quantized = []
        #print(f"inside   {x_id.shape}")
        for i, part in enumerate(self.partSeg):
            x_current = self.quantizers[i].forward_from_code_idx(x_id[..., i])
            x_quantized.append(x_current)
        return x_quantized

    # ========== 从 quantized feature list 直接解码 ==========
    def forward_decoder_from_quantized_codes(self, x_quantized):
        x_quantized = torch.cat(x_quantized, dim = 1) #从6个list变成b 128*6 49了
        x_decoder = self.decoder(x_quantized) # 输出 b,263,196
        x_out = x_decoder.unsqueeze(2)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
        norm=None):
        super().__init__()

        default_input_dim = 251 if getattr(args, "dataname", "t2m") == "kit" else 263
        self.input_dim = int(getattr(args, "input_dim", default_input_dim))
        self.nb_joints = 21 if self.input_dim == 251 else 22
        self.vqvae = VQVAE_251(
            args, nb_code, code_dim, code_dim,
            down_t, stride_t, width, depth,
            dilation_growth_rate,
            activation=activation, norm=norm
        )

    def forward(self, x, type='full'):
        """
        type: ['full', 'encode', 'decode']
        """
        if type == 'full':
            x_out, loss, perplexity = self.vqvae(x)
            return x_out, loss, perplexity
        elif type == 'encode':
            quants = self.vqvae.encode(x)  # (B, T_codes)
            return quants
        elif type == 'decode':
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')
    
    def get_code_idx(self, x):
        return self.vqvae.get_quantized_codes(x)
        # return self.vqvae.get_quantized_codes(x)
