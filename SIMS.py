import copy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class SIMS:
    def __init__(
        self,
        model,
        params,
        device,
        local_rank=None,
        scheduler_builder=None,
        inner_weight_decay=1e-6,
        inner_lr=None
    ):
        self.device = device
        self.local_rank = local_rank
        self.use_quad = bool(params.use_quad)
        self.tau = params.tau
        self.lmbd = params.lmbd
        self.eps = 1e-8

        self.y_model = copy.deepcopy(self._unwrap(model)).to(device)
        if isinstance(model, DDP):
            self.y_model = DDP(
                self.y_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        self.y_optimizer = torch.optim.Adam(
            (p for p in self.y_model.parameters() if p.requires_grad),
            lr=params.lr if inner_lr is None else inner_lr,
            weight_decay=inner_weight_decay,
        )
        if scheduler_builder is None:
            self.y_scheduler = None
        else:
            self.y_scheduler = scheduler_builder(self.y_optimizer, params)

    @staticmethod
    def _unwrap(model):
        return model.module if isinstance(model, DDP) else model

    @staticmethod
    def _stack_losses(loss_fn, model, batch):
        return torch.stack(loss_fn(model, *batch))

    def _trainable_param_dict(self, model):
        return {
            name: param.detach()
            for name, param in self._unwrap(model).named_parameters()
            if param.requires_grad
        }

    def _add_quad_grad_to_y(self, x_param_dict):
        with torch.no_grad():
            for name, param_y in self._unwrap(self.y_model).named_parameters():
                if not param_y.requires_grad or name not in x_param_dict:
                    continue
                reg_grad = self.lmbd * (
                    param_y - x_param_dict[name]
                )
                if param_y.grad is None:
                    param_y.grad = reg_grad.detach().clone()
                else:
                    param_y.grad.add_(reg_grad)

    def _add_quad_grad_to_x(self, model):
        y_param_dict = {
            name: param.detach()
            for name, param in self._unwrap(self.y_model).named_parameters()
        }
        with torch.no_grad():
            for name, param_x in self._unwrap(model).named_parameters():
                if not param_x.requires_grad or name not in y_param_dict:
                    continue
                reg_grad = self.lmbd * (y_param_dict[name] - param_x)
                if param_x.grad is None:
                    param_x.grad = reg_grad.detach().clone()
                else:
                    param_x.grad.add_(reg_grad)
                

    def backward(
        self,
        model,
        loss_fn,
        batch,
        accumulate_step=1,
    ):
        with torch.no_grad():
            fx_const = self._stack_losses(loss_fn, model, batch)

        self.y_model.train()
        if self.use_quad:
            x_param_dict = self._trainable_param_dict(model)
        else:
            x_param_dict = None

        self.y_optimizer.zero_grad()
        fy = self._stack_losses(loss_fn, self.y_model, batch)
        h = self.tau * torch.logsumexp(
            (torch.log(fy + self.eps) - torch.log(fx_const + self.eps)) / self.tau,
            dim=0,
        )
        h.backward()

        if self.use_quad:
            self._add_quad_grad_to_y(x_param_dict)

        self.y_optimizer.step()
        if self.y_scheduler is not None:
            self.y_scheduler.step()

        fx = self._stack_losses(loss_fn, model, batch)

        with torch.no_grad():
            fy_final = self._stack_losses(loss_fn, self.y_model, batch)

        h_xy = self.tau * torch.logsumexp(
            (torch.log(fy_final + self.eps) - torch.log(fx + self.eps)) / self.tau,
            dim=0,
        )
        v_est = -h_xy
        (v_est / accumulate_step).backward()

        if self.use_quad:
            self._add_quad_grad_to_x(model)

        return fx.detach(), fy_final.detach(), v_est.detach()
