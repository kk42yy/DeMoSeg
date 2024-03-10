import torch
from torch import nn

from training.trainer.BaseTrainer import BasicTrainer
from training.network.DeMoSeg import DeMoSeg

class DeMoSeg_Trainer(BasicTrainer):
    def __init__(self, task) -> None:
        self.task = task
        self.MaxEpoch = 1000
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5

        self.initialze()

    def initialze(self, training=True):
        super().initialze(training)
        self.network = DeMoSeg(input_channels=4, num_classes=4, num_pool=5)
        self.network.cuda()
        self.network.inference_apply_nonlin = lambda x: torch.nn.functional.softmax(x,1)
        self.network.training_apply_nonlin = lambda x: x
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        
    def run_train(self):
        pass