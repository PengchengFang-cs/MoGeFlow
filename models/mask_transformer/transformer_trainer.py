import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0


def build_lr_scheduler(optimizer, opt, total_iters):
    scheduler_name = getattr(opt, 'lr_scheduler', 'multistep')

    if scheduler_name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opt.milestones,
            gamma=opt.gamma,
        )

    if scheduler_name == 'half_cosine':
        cosine_iters = max(1, total_iters - max(0, opt.warm_up_iter))
        eta_min = opt.lr * opt.eta_min_ratio
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_iters,
            eta_min=eta_min,
        )

    raise ValueError(f'Unsupported lr_scheduler: {scheduler_name}')


def should_step_scheduler(opt, it):
    scheduler_name = getattr(opt, 'lr_scheduler', 'multistep')

    if scheduler_name == 'multistep':
        return True

    if scheduler_name == 'half_cosine':
        return it >= opt.warm_up_iter

    raise ValueError(f'Unsupported lr_scheduler: {scheduler_name}')

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)

        return _loss, _acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'lr_scheduler_type': getattr(self.opt, 'lr_scheduler', 'multistep'),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            ckpt_scheduler_type = checkpoint.get('lr_scheduler_type', 'multistep')
            if ckpt_scheduler_type == getattr(self.opt, 'lr_scheduler', 'multistep'):
                self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
            else:
                print(f'Skip scheduler resume: checkpoint uses {ckpt_scheduler_type}, current run uses {self.opt.lr_scheduler}')
        except Exception as exc:
            print(f'Resume wo optimizer: {exc}')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        epoch = 0
        it = 0

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        self.scheduler = build_lr_scheduler(self.opt_t2m_transformer, self.opt, total_iters)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        if self.opt.lr_scheduler == 'half_cosine':
            print(f'LR Scheduler: half_cosine, warmup_iters={self.opt.warm_up_iter}, eta_min={self.opt.lr * self.opt.eta_min_ratio:.8f}')
        else:
            print(f'LR Scheduler: multistep, warmup_iters={self.opt.warm_up_iter}, milestones={self.opt.milestones}, gamma={self.opt.gamma}')
        eval_enabled = (eval_val_loader is not None) and (eval_wrapper is not None)
        print(f'Eval starts from epoch {getattr(self.opt, "eval_start_epoch", 0)} and runs every {self.opt.eval_every_e} epochs')
        if eval_enabled:
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        else:
            print('Iters Per Epoch, Training: %04d, Validation: %03d (skip_eval=True)' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
        if eval_enabled and not getattr(self.opt, 'skip_eval', False) and epoch >= getattr(self.opt, 'eval_start_epoch', 0):
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_mask_transformer(
                self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
                best_fid=100, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=100, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=False, save_anim=False
            )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                if should_step_scheduler(self.opt, it):
                    self.scheduler.step()
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            run_eval = (epoch >= getattr(self.opt, 'eval_start_epoch', 0)) and (epoch % self.opt.eval_every_e == 0)
            if not run_eval:
                continue
            if getattr(self.opt, 'skip_eval', False):
                continue

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            if len(val_loss) > 0:
                val_loss_mean = np.mean(val_loss)
                val_acc_mean = np.mean(val_acc)
                print(f"Validation loss:{val_loss_mean:.3f}, accuracy:{val_acc_mean:.3f}")

                self.logger.add_scalar('Val/loss', val_loss_mean, epoch)
                self.logger.add_scalar('Val/acc', val_acc_mean, epoch)

                if val_acc_mean > best_acc:
                    print(f"Improved accuracy from {best_acc:.02f} to {val_acc_mean}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                    best_acc = val_acc_mean
            else:
                print("Validation skipped: val_loader is empty.")

            if eval_enabled:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_mask_transformer(
                    self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=True
                )


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model, base_transformer=None):
        if base_transformer is not None:
            raise ValueError('Base-pred residual training has been retired; use GT coarse tokens (source MoMask style).')
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            # self.l1_criterion = torch.nn.SmoothL1Loss()


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'lr_scheduler_type': getattr(self.opt, 'lr_scheduler', 'multistep'),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            ckpt_scheduler_type = checkpoint.get('lr_scheduler_type', 'multistep')
            if ckpt_scheduler_type == getattr(self.opt, 'lr_scheduler', 'multistep'):
                self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
            else:
                print(f'Skip scheduler resume: checkpoint uses {ckpt_scheduler_type}, current run uses {self.opt.lr_scheduler}')
        except Exception as exc:
            print(f'Resume wo optimizer: {exc}')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        epoch = 0
        it = 0

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        self.scheduler = build_lr_scheduler(self.opt_res_transformer, self.opt, total_iters)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        if self.opt.lr_scheduler == 'half_cosine':
            print(f'LR Scheduler: half_cosine, warmup_iters={self.opt.warm_up_iter}, eta_min={self.opt.lr * self.opt.eta_min_ratio:.8f}')
        else:
            print(f'LR Scheduler: multistep, warmup_iters={self.opt.warm_up_iter}, milestones={self.opt.milestones}, gamma={self.opt.gamma}')
        eval_enabled = (eval_val_loader is not None) and (eval_wrapper is not None)
        print(f'Eval starts from epoch {getattr(self.opt, "eval_start_epoch", 0)} and runs every {self.opt.eval_every_e} epochs')
        if eval_enabled:
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        else:
            print('Iters Per Epoch, Training: %04d, Validation: %03d (skip_eval=True)' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = 100, 100, 0, 0, 0, 100
        if eval_enabled and not getattr(self.opt, 'skip_eval', False) and epoch >= getattr(self.opt, 'eval_start_epoch', 0):
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_res_transformer(
                self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch,
                best_fid=100, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=100, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=False, save_anim=False,
            )
        best_loss = 100
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                if should_step_scheduler(self.opt, it):
                    self.scheduler.step()
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            run_eval = (epoch >= getattr(self.opt, 'eval_start_epoch', 0)) and (epoch % self.opt.eval_every_e == 0)
            if not run_eval:
                continue
            if getattr(self.opt, 'skip_eval', False):
                continue

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            if len(val_loss) > 0:
                val_loss_mean = np.mean(val_loss)
                val_acc_mean = np.mean(val_acc)
                print(f"Validation loss:{val_loss_mean:.3f}, Accuracy:{val_acc_mean:.3f}")

                self.logger.add_scalar('Val/loss', val_loss_mean, epoch)
                self.logger.add_scalar('Val/acc', val_acc_mean, epoch)

                if val_loss_mean < best_loss:
                    print(f"Improved loss from {best_loss:.02f} to {val_loss_mean}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                    best_loss = val_loss_mean

                if val_acc_mean > best_acc:
                    print(f"Improved acc from {best_acc:.02f} to {val_acc_mean}!!!")
                    # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                    best_acc = val_acc_mean
            else:
                print("Validation skipped: val_loader is empty.")

            if eval_enabled:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_res_transformer(
                    self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=True,
                )
