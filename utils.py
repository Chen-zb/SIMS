import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch import optim
import math

mask = None

class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """
    def __init__(self, dataset='NYUD', pos_weight=None):
        super().__init__()
        ignore_index = 255
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label, reduction='mean'):
        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return 0
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction=reduction)
        loss /= factor
        return loss

def model_fit(x_pred, x_output, task_type):
    device = x_pred.device
    
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type in ['semantic', 'segment_semantic']:
        # semantic loss: depth-wise cross entropy
        ignore_index = -1
        loss = F.nll_loss(x_pred, x_output, ignore_index=ignore_index)

    if task_type in ['depth', 'depth_zbuffer', 'keypoints2d', 'edge_texture']:
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    
    if task_type == 'edge':
        loss = BalancedBinaryCrossEntropyLoss(pos_weight=0.95)(x_pred, x_output)

    return loss

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
            with torch.no_grad():
                k = (target >= 0) & (target < n)
                inds = n * target[k].to(torch.int64) + pred[k]
                self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        with torch.no_grad():
            h = self.mat.float()
            acc = torch.diag(h).sum() / h.sum()
            iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
            return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output, dataset='NYUv2'):
    with torch.no_grad():
        device = x_pred.device
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        if mask is not None:
            binary_mask *= (mask.int() == 1)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
               (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output, dataset='NYUv2'):
    with torch.no_grad():
        binary_mask = (torch.sum(x_output, dim=1) != 0)
        if mask is not None:
            binary_mask *= (mask[:,0,:,:].int() == 1)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1))#.detach().cpu().numpy()
    #     error = np.degrees(error)
        error = torch.rad2deg(error)
        return torch.mean(error).item(), torch.median(error).item(), \
               torch.mean((error < 11.25)*1.0).item(), torch.mean((error < 22.5)*1.0).item(), \
               torch.mean((error < 30)*1.0).item()

class EdgeMeter(object):
    def __init__(self, pos_weight):
        self.loss = 0
        self.n = 0
        self.loss_function = BalancedBinaryCrossEntropyLoss(pos_weight=pos_weight)
        self.ignore_index = self.loss_function.ignore_index
        
    def update(self, pred, gt):
        gt = gt.squeeze()
        pred = pred.squeeze()
        valid_mask = (gt != self.ignore_index)
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self, verbose=True):
        eval_dict = {'loss': self.loss / self.n}

        if verbose:
            print('\n Edge Detection Evaluation')
            print('Edge Detection Loss %.3f' %(eval_dict['loss']))

        return eval_dict


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
            with torch.no_grad():
                k = (target >= 0) & (target < n)
                inds = n * target[k].to(torch.int64) + pred[k]
                self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        with torch.no_grad():
            h = self.mat.float()
            acc = torch.diag(h).sum() / h.sum()
            iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
            return torch.mean(iu).item(), acc.item()

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer : torch.optim.Optimizer,
        max_lr : float = 0.1,
        min_lr : float = 0.0,
        warmup_steps : int = 0,
        max_steps : int = 1,
        alpha : float = 0.,
        last_epoch : int = -1
    ):
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        
        self.alpha = alpha # decrease rate of max learning rate by cycle
        self.max_steps = max_steps
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.max_lr * self.last_epoch / self.warmup_steps
            return curr_lr
        else:
            _step = min(self.last_epoch, self.max_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * _step / self.max_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.max_lr * decayed # learning_rate * decayed

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: 
            param_group['lr'] = _lr


class CyclicScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        interval_steps = [],
        interval_lrs = [],
        last_epoch = -1,
    ):        
        self.optimizer = optimizer

        self.interval_steps = interval_steps
        self.interval_lrs = interval_lrs

        self.last_epoch = last_epoch

        super(CyclicScheduler, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.interval_lrs[0]
    
    def get_lr(self):
        for _i in range(0, len(self.interval_steps)-1):
            if self.last_epoch >= self.interval_steps[_i] and self.last_epoch < self.interval_steps[_i + 1]:
                _alpha = (self.last_epoch - self.interval_steps[_i]) / (self.interval_steps[_i + 1] - self.interval_steps[_i] + 1e-6)
                if _alpha < 0:
                    _alpha = 0
                if _alpha >= 1:
                    _alpha = 1
                curr_lr = _alpha * self.interval_lrs[_i + 1] + (1.0 - _alpha) * self.interval_lrs[_i]             
                return curr_lr
        return self.interval_lrs[-1]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        #self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: #, self.get_lr()):
            param_group['lr'] = _lr



def get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def create_optimizer_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            max_lr=args.lr, 
            min_lr=0.0, 
            warmup_steps=args.warmup_step, 
            max_steps=args.max_step, alpha=0
        )
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    elif args.scheduler == 'cycle':
        if args.i_steps is not None:
            args.i_steps = [int(_i) for _i in args.i_steps.split(',')]
            args.i_lrs = [float(_i) for _i in args.i_lrs.split(',')]
        args.max_step = args.i_steps[-1]
        print('max_step is rest to', args.max_step)
        scheduler = CyclicScheduler(
            optimizer, interval_steps=args.i_steps, interval_lrs=args.i_lrs
        )
    elif args.scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    elif args.scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.max_step / 4, gamma=0.5)
    else:
        # constant leanring rate.
        scheduler = None
    return scheduler


def save_model_pred_for_one_task(sample, output, save_dirs):
    """ Save model predictions for one task"""

    inputs, fnames = sample['image'].cuda(non_blocking=True), sample['image_name']
    fnames = fnames.tolist()
    output_task = output

    for jj in range(int(inputs.size()[0])):
        if len(sample['image'][jj].unique()) == 1 and sample['image'][jj].unique() == 255.:
            continue
        fname = fnames[jj]
        im_height = inputs[jj].shape[-2]
        im_width = inputs[jj].shape[-1]
        pred = output_task[jj].squeeze() # (H, W) or (H, W, C)
        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[:2]:
            delta_height = max(pred.shape[0] - im_height, 0)
            delta_width = max(pred.shape[1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                height_begin = torch.div(delta_height, 2, rounding_mode="trunc")
                height_location = [height_begin, height_begin + im_height]
                width_begin =torch.div(delta_width, 2, rounding_mode="trunc")
                width_location = [width_begin, width_begin + im_width]
                pred = pred[height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[:2] == (im_height, im_width)
        if pred.ndim == 3:
            raise
        result = pred.cpu().numpy()
        np.save(os.path.join(save_dirs, str(fname) + '.npy'), result)
