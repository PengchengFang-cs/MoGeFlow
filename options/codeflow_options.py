import argparse
from pathlib import Path


class TrainCodeFlowOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self) -> None:
        p = self.parser
        p.add_argument("--name", type=str, default="codeflow_tconcat_dit_b")
        p.add_argument("--dataset_name", type=str, default="t2m", choices=["t2m", "kit"])
        p.add_argument("--data_root", type=str, default="dataset/HumanML3D")
        p.add_argument("--output_dir", type=str, default="./checkpoints/t2m/codeflow_tconcat_dit_b")
        p.add_argument("--kv_root", type=str, default=".")
        p.add_argument("--dataset_opt_path", type=str, default="")
        p.add_argument("--vq_backend", type=str, default="kv_part", choices=["kv_part", "momask_rvq"])
        p.add_argument("--vq_checkpoint", type=str, default="")
        p.add_argument("--vq_partition", type=str, default="")
        p.add_argument("--vq_opt_path", type=str, default="")
        p.add_argument("--mean_path", type=str, default="")
        p.add_argument("--std_path", type=str, default="")
        p.add_argument("--clip_path", type=str, default="")
        p.add_argument("--clip_version", type=str, default="ViT-B/32")

        p.add_argument("--representation", type=str, default="t_concat", choices=["t_concat", "part_structured"])
        p.add_argument("--coupling_mode", type=str, default="holder_query", choices=["holder_query", "frame_grouped"])
        p.add_argument("--holder_depth", type=int, default=2)
        p.add_argument("--holder_mlp_ratio", type=float, default=4.0)
        p.add_argument("--code_dim", type=int, default=128)
        p.add_argument("--num_parts", type=int, default=6)
        p.add_argument("--num_codes", type=int, default=128)
        p.add_argument(
            "--part_hidden_dim",
            type=int,
            default=0,
            help="Per-part/RVQ-layer hidden width before frame grouping. 0 means use code_dim.",
        )
        p.add_argument("--hidden_size", type=int, default=768)
        p.add_argument("--num_heads", type=int, default=12)
        p.add_argument("--depth_double", type=int, default=6)
        p.add_argument("--depth_single", type=int, default=12)
        p.add_argument("--mlp_ratio", type=float, default=4.0)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--time_patch", type=int, default=1, help="Holder-query CodeFlow uses one latent timestep per patch.")
        p.add_argument("--terminal_mode", type=str, default="tied_logits", choices=["nearest", "tied_logits", "learned_head"])
        p.add_argument("--terminal_tau", type=float, default=1.0)
        p.add_argument("--terminal_tau_mode", type=str, default="fixed", choices=["fixed", "codebook_nn"])
        p.add_argument("--terminal_tau_floor", type=float, default=1e-6)
        p.add_argument("--code_ce_t_min", type=float, default=0.0)
        p.add_argument("--code_ce_t_max", type=float, default=1.0)
        p.add_argument("--code_ce_gamma", type=float, default=0.0)
        p.add_argument("--code_ce_normalize", action="store_true")
        p.add_argument("--flow_loss_weight", type=float, default=1.0)
        p.add_argument("--terminal_loss_weight", type=float, default=1.0)
        p.add_argument("--clean_loss_weight", type=float, default=0.0)

        p.add_argument("--motion_length", type=int, default=196)
        p.add_argument("--unit_length", type=int, default=4)
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--max_epoch", type=int, default=600)
        p.add_argument("--max_steps", type=int, default=0)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--lr_scheduler", type=str, default="multistep", choices=["multistep", "half_cosine"])
        p.add_argument("--gamma", type=float, default=0.5, help="Learning rate decay factor for multistep scheduler.")
        p.add_argument(
            "--milestone_unit",
            type=str,
            default="auto",
            choices=["auto", "iter", "epoch", "epoch_ratio"],
            help="How to interpret multistep milestones. auto: (0,1] -> epoch ratio, integer <= max_epoch -> epoch.",
        )
        p.add_argument(
            "--milestones",
            default=[0.8, 0.9, 0.95],
            nargs="+",
            type=float,
            help="Multistep LR milestones. In auto mode, values in (0,1] are treated as epoch ratios.",
        )
        p.add_argument("--eta_min_ratio", type=float, default=0.1, help="Minimum LR ratio for half_cosine scheduler.")
        p.add_argument("--weight_decay", type=float, default=0.01)
        p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
        p.add_argument("--warmup_steps", type=int, default=2000)
        p.add_argument("--grad_clip", type=float, default=1.0)
        p.add_argument("--amp", action="store_true")
        p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
        p.add_argument("--seed", type=int, default=3407)

        p.add_argument("--cond_drop_prob", type=float, default=0.1)
        p.add_argument("--self_cond_prob", type=float, default=0.5)
        p.add_argument("--disable_self_condition", action="store_true")
        p.add_argument("--time_schedule", type=str, default="logit_normal", choices=["logit_normal", "uniform"])
        p.add_argument("--denoiser_p_mean", type=float, default=-1.5)
        p.add_argument("--denoiser_p_std", type=float, default=0.8)
        p.add_argument("--noise_scale", type=float, default=1.0)
        p.add_argument("--latent_norm_mode", type=str, default="none", choices=["none", "codebook"])
        p.add_argument("--latent_offset", type=float, default=0.0)
        p.add_argument("--latent_norm_eps", type=float, default=1e-6)
        p.add_argument("--sampling_schedule", type=str, default="uniform", choices=["uniform", "logit_normal"])
        p.add_argument("--sampling_method", type=str, default="ode", choices=["ode", "sde"])
        p.add_argument("--sde_gamma", type=float, default=0.0)
        p.add_argument("--decode_mode", type=str, default="nearest", choices=["nearest", "ids", "continuous"])

        p.add_argument("--log_every", type=int, default=50)
        p.add_argument("--save_every", type=int, default=0, help="Deprecated; training now keeps latest and top-k best checkpoints.")
        p.add_argument("--best_checkpoint_limit", type=int, default=5, help="Number of best FID/Top3 checkpoints to retain.")
        p.add_argument("--full_eval_every_epoch", type=int, default=5)
        p.add_argument("--full_eval_start_epoch", type=int, default=0)
        p.add_argument("--full_eval_batch_size", type=int, default=32)
        p.add_argument("--full_eval_num_workers", type=int, default=4)
        p.add_argument("--full_eval_steps", type=int, default=32)
        p.add_argument("--full_eval_cond_scale", type=float, default=3.0)
        p.add_argument("--full_eval_repeat_times", type=int, default=1)
        p.add_argument(
            "--full_eval_seed",
            type=int,
            default=3407,
            help="Fixed base seed for full HumanML3D eval during training. Repeat i uses full_eval_seed+i.",
        )
        p.add_argument("--disable_full_eval_ema", action="store_true")
        p.add_argument("--geometry_severe_quantile", type=float, default=0.75)
        p.add_argument("--allow_non_vq_stats", action="store_true")
        p.add_argument("--disable_vq_contract_check", action="store_true")
        p.add_argument("--vq_contract_samples", type=int, default=2)
        p.add_argument("--grad_health_check_steps", type=int, default=20)
        p.add_argument("--grad_health_min_abs", type=float, default=0.0)
        p.add_argument("--disable_grad_health_check", action="store_true")
        p.add_argument("--resume", type=str, default="")
        p.add_argument("--gpu_id", type=int, default=-1)

    def parse(self):
        opt = self.parser.parse_args()
        kv_root = Path(opt.kv_root).expanduser()
        repo_root = Path(__file__).resolve().parents[1]
        momask_root = repo_root / "checkpoints" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2"
        if opt.vq_backend == "momask_rvq":
            if not opt.vq_checkpoint:
                opt.vq_checkpoint = str(momask_root / "model" / "net_best_fid.tar")
            if not opt.vq_opt_path:
                opt.vq_opt_path = str(momask_root / "opt.txt")
            if not opt.mean_path:
                opt.mean_path = str(momask_root / "meta" / "mean.npy")
            if not opt.std_path:
                opt.std_path = str(momask_root / "meta" / "std.npy")
        else:
            if not opt.vq_checkpoint:
                opt.vq_checkpoint = str(kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth")
            if not opt.vq_partition:
                opt.vq_partition = str(kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json")
            if not opt.mean_path:
                opt.mean_path = str(kv_root / "checkpoints" / "stats" / "mean.npy")
            if not opt.std_path:
                opt.std_path = str(kv_root / "checkpoints" / "stats" / "std.npy")
        if not opt.clip_path:
            opt.clip_path = str(kv_root / "checkpoints" / "clip" / "ViT-B-32.pt")
        return opt
