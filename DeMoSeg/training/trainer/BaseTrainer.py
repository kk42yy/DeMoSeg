import torch
from torch import nn

class BasicTrainer(object):
    def __init__(self, task) -> None:
        self.task = task
        self.MaxEpoch = 1000
        self.initial_lr = 1e-2
        self.pin_memory = True
        self.plans = None

        self.initialze()

    def get_DA(self):
        pass

    def get_dataloader(self):
        pass

    def get_loss(self):
        pass

    def initialze(self, training=True):
        self.process_plans(self.plans)
        self.get_DA()
        if training:
            self.tr_dataloader, self.vl_dataloader = self.get_dataloader()

    def run_train(self):
        pass

    def validation(self):
        pass