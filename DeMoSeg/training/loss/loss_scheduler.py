import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class DeepSupervisionLoss(_Loss):
    def __init__(self, loss, weights=[1.]):
        super().__init__()
        self.weight_factors = tuple(weights)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        
        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        for i, inputs in enumerate(zip(*args)):
            if i == 0:
                l = weights[0] * self.loss(*inputs)
            else:
                if weights[i] != 0.0:
                    l += weights[i] * self.loss(*inputs)
        return l
