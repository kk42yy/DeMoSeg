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

class DeepSupervisionLoss_noKL(_Loss):
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

class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(DeepSupervisionLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        if self.training:
            x, kl = x
            
        else:
            kl = 0

        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])

        return l + kl
