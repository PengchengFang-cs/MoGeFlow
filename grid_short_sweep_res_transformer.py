import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from os.path import join as pjoin

from data.t2m_dataset import Text2MotionDataset
from short_sweep_res_transformer import (
    build_loader,
    build_parser,
    configure_dataset,
    make_subset_indices,
    run_short_trial,
)
from utils.fixseed import fixseed


def add_grid_args(parser):
    parser.description = "Sequential short-run sweep for residual transformer hyperparameters."
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to sweep. Defaults to the single --batch_size value.",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=None,
        help="Learning rates to sweep. Defaults to the single --lr value.",
    )
    parser.add_argument(
        "--res_uni_path_as",
        type=float,
        nargs="+",
        default=None,
        help="Uni-mask path exponent values to sweep. Defaults to the single --res_uni_path_a value.",
    )
    parser.add_argument(
        "--res_uni_path_cs",
        type=float,
        nargs="+",
        default=None,
        help="Uni-mask path coefficient values to sweep. Defaults to the single --res_uni_path_c value.",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=0,
        help="Optional cap on the number of grid trials. 0 means run all combinations.",
    )
    parser.add_argument(
        "--save_grid_json",
        type=str,
        default="",
        help="Optional path to save the list of completed grid results as JSON.",
    )
    return parser


def save_results(results, save_path):
    if not save_path:
        return
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))


def main():
    parser = add_grid_args(build_parser())
    opt = parser.parse_args()
    opt.is_train = False
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    batch_sizes = opt.batch_sizes or [opt.batch_size]
    lrs = opt.lrs or [opt.lr]
    path_as = opt.res_uni_path_as or [opt.res_uni_path_a]
    path_cs = opt.res_uni_path_cs or [opt.res_uni_path_c]

    grid = list(itertools.product(batch_sizes, lrs, path_as, path_cs))
    if opt.max_trials > 0:
        grid = grid[: opt.max_trials]
    if not grid:
        raise ValueError("Grid is empty. Provide at least one batch size / lr / uni-mask value.")

    fixseed(opt.seed)
    _, train_split = configure_dataset(opt)
    val_split = pjoin(opt.data_root, "val.txt")

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, "meta", "mean.npy"))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, "meta", "std.npy"))

    print("Loading datasets once for grid short sweep...")
    train_dataset = Text2MotionDataset(opt, mean.copy(), std.copy(), train_split)
    val_dataset = Text2MotionDataset(opt, mean.copy(), std.copy(), val_split)

    train_indices = make_subset_indices(len(train_dataset), opt.train_subset_size, opt.subset_seed)
    val_indices = make_subset_indices(len(val_dataset), opt.val_subset_size, opt.subset_seed + 1)

    results = []
    for trial_idx, (batch_size, lr, path_a, path_c) in enumerate(grid):
        if len(train_indices) < batch_size:
            raise ValueError(
                f"train_subset_size={len(train_indices)} is smaller than batch_size={batch_size}."
            )
        trial_opt = copy.deepcopy(opt)
        trial_opt.batch_size = batch_size
        trial_opt.lr = lr
        trial_opt.res_uni_path_a = path_a
        trial_opt.res_uni_path_c = path_c
        trial_opt.trial_seed_offset = opt.trial_seed_offset + trial_idx * 1000

        print(
            f"Running trial {trial_idx + 1}/{len(grid)}: "
            f"bs={batch_size}, lr={lr}, path_a={path_a}, path_c={path_c}"
        )
        train_loader = build_loader(
            dataset=train_dataset,
            indices=train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=trial_opt.seed,
            num_workers=trial_opt.num_workers,
        )
        val_loader = build_loader(
            dataset=val_dataset,
            indices=val_indices,
            batch_size=trial_opt.val_batch_size,
            shuffle=False,
            seed=trial_opt.seed + 1,
            num_workers=trial_opt.num_workers,
        )

        result = run_short_trial(
            opt=trial_opt,
            train_loader=train_loader,
            val_loader=val_loader,
            full_train_size=len(train_dataset),
        )
        result["trial_index"] = trial_idx
        results.append(result)
        save_results(results, opt.save_grid_json)
        print("GRID_SHORT_SWEEP_RESULT " + json.dumps(result, sort_keys=True))

    if opt.save_json:
        save_results(results[-1:], opt.save_json)


if __name__ == "__main__":
    main()
