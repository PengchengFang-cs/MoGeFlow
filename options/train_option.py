from options.base_option import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--lr_scheduler', type=str, default='multistep',
                                 choices=['half_cosine', 'multistep'],
                                 help='Learning rate scheduler type')
        self.parser.add_argument('--gamma', type=float, default=0.1,
                                 help='Learning rate decay factor for multistep scheduler')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="Multistep learning rate schedule milestones (iterations)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
        self.parser.add_argument('--eta_min_ratio', type=float, default=0.1,
                                 help='Minimum LR as a ratio of peak LR for half-cosine scheduler')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')
        self.parser.add_argument('--res_train_with_base_pred', action='store_true',
                                 help='Train residual transformer using frozen base predictions instead of GT coarse tokens.')
        self.parser.add_argument('--res_train_base_opt_path', type=str, default='',
                                 help='Path to the frozen base model opt.txt used for residual training/eval.')
        self.parser.add_argument('--res_train_base_model_path', type=str, default='',
                                 help='Path to the frozen base model checkpoint used for residual training/eval.')
        self.parser.add_argument('--res_train_base_time_steps', type=int, default=18,
                                 help='Sampling steps used by the frozen base model when generating coarse tokens.')
        self.parser.add_argument('--res_train_base_cond_scale', type=float, default=4.0,
                                 help='CFG scale used by the frozen base model when generating coarse tokens.')
        self.parser.add_argument('--res_train_base_topkr', type=float, default=0.9,
                                 help='Top-k filter threshold used by the frozen base model when generating coarse tokens.')
        self.parser.add_argument('--res_train_base_gsample', action='store_true',
                                 help='Use Gumbel sampling when the frozen base model generates coarse tokens.')
        self.parser.add_argument('--res_use_uni_mask', action='store_true',
                                 help='Enable FUDOKI-style uni-mask corruption in residual transformer training.')
        self.parser.add_argument('--res_uni_solver_steps', type=int, default=16,
                                 help='Number of iterative refinement steps for FUDOKI-style residual generation.')
        self.parser.add_argument('--res_uni_path_a', type=float, default=0.9,
                                 help='FUDOKI metric-path exponent a in beta_t = c * (t/(1-t))^a.')
        self.parser.add_argument('--res_uni_path_c', type=float, default=3.0,
                                 help='FUDOKI metric-path coefficient c in beta_t = c * (t/(1-t))^a.')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        # self.parser.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--eval_start_epoch', type=int, default=0,
                                 help='Start running validation/evaluation only from this epoch.')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')


        self.is_train = True


class TrainLenEstOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        # args = vars(self.opt)
        return self.opt
