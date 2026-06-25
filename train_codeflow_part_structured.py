"""Train Part-Structured Code Flow."""

from models.codeflow import PartStructuredMotionCodeFlow
from models.codeflow.trainer import main


if __name__ == "__main__":
    main(
        model_cls=PartStructuredMotionCodeFlow,
        option_defaults={
            "name": "codeflow_part_structured_pscf",
            "output_dir": "./checkpoints/t2m/codeflow_part_structured_pscf",
            "representation": "part_structured",
            "coupling_mode": "frame_grouped",
            "flow_loss_weight": 1.0,
            "terminal_loss_weight": 1.0,
            "clean_loss_weight": 0.0,
            "terminal_mode": "tied_logits",
            "terminal_tau_mode": "codebook_nn",
            "code_ce_t_min": 0.35,
            "code_ce_t_max": 0.90,
            "code_ce_gamma": 2.0,
            "code_ce_normalize": True,
            "disable_self_condition": True,
            "latent_norm_mode": "codebook",
            "latent_offset": 0.0,
            "latent_norm_eps": 1e-6,
            "noise_scale": 1.0,
            "time_schedule": "uniform",
            "sampling_schedule": "uniform",
            "sampling_method": "ode",
            "sde_gamma": 0.0,
            "decode_mode": "nearest",
            "full_eval_cond_scale": 3.0,
        },
    )
