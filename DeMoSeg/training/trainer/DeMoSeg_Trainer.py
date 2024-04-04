import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from time import time
from torch import nn

from training.trainer.BaseTrainer import BasicTrainer
from training.network.DeMoSeg import DeMoSeg
from training.dataloader.MultiThreadDataLoader import MultiThreadedDataLoader
from training.deepsupervision.deepsupervision_utils import Downsample_Label_for_DeepSupervision

class DeMoSeg_Trainer(BasicTrainer):
    def __init__(self, task='2020', fold=0, basepath='') -> None:
        super().__init__(task, fold, basepath)
        self.MaxEpoch = 1000
        self.initial_lr = 1e-2
        self.Iteration_for_trainingepoch = 250
        self.Iteration_for_validationepoch = 50

    def get_network(self):
        num_classes = {
            '2020': 4,
            '2018': 4,
            '2015': 5
        }
        self.network = DeMoSeg(input_channels=4, num_classes=num_classes[self.task], num_pool=5)
        self.network.inference_apply_nonlin = torch.nn.Softmax(dim=1)
        self.network.training_apply_nonlin = nn.Identity()
        self.seg_down_sampling = Downsample_Label_for_DeepSupervision(
            ds_scales=[[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(np.vstack([[2]*3 for _ in range(self.network.num_pool)]), axis=0))[:-1],
            order=0
        )

    def get_dataloader(self):
        self.train_loader = MultiThreadedDataLoader(
            data_list=self.Tr_list,
            batch_size = self.batch_size,
            transform=self.train_transform,
            num_processes=18,
            num_cached=6,
            seeds=None,
            pin_memory=self.pin_memory,
            wait_time=0.02
        )
        self.val_loader = MultiThreadedDataLoader(
            data_list=self.Val_list,
            batch_size = self.batch_size,
            transform=self.val_transform,
            num_processes=9,
            num_cached=3,
            seeds=None,
            pin_memory=self.pin_memory,
            wait_time=0.02
        )
    
    def run_train(self):
        
        # start training
        torch.cuda.empty_cache()
        self.network.to(self.device)
        self.network.do_ds = self.deep_supervision
        self.get_dataloader()
        
        # epoch starting
        for epoch in range(self.MaxEpoch):
            self.network.train()
            self.lr_scheduler.step(epoch)
            self.logger.info("")
            self.logger.info(f"Epoch: {epoch}, current lr: {np.round(self.optimizer.param_groups[0]['lr'], decimals=6)}")

            # batch training iteration
            epoch_loss, epoch_startime = np.zeros(self.Iteration_for_trainingepoch), time()
            for idx in tqdm(range(self.Iteration_for_trainingepoch)):
                epoch_loss[idx] = self.training_epoch(next(self.train_loader))

            # batch validation iteration
            with torch.no_grad():
                self.network.eval()
                val_loss, val_dice = np.zeros(self.Iteration_for_validationepoch), np.zeros((self.Iteration_for_validationepoch, self.network.num_classes-1))
                for idx in tqdm(range(self.Iteration_for_validationepoch)):
                    val_loss[idx], val_dice[idx][:] = self.validation_epoch(next(self.val_loader))

            self.logger.info(f"train_loss: {np.round(np.nanmean(epoch_loss), 4)}")
            self.logger.info(f"val_loss: {np.round(np.nanmean(val_loss), 4)}")
            self.logger.info(f"pseudo val_dice: {np.round(np.nanmean(val_dice, axis=0), 4)}, mean: {np.round(np.mean(val_dice), 4)}")
            self.logger.info(f"epoch time: {np.round(time()-epoch_startime, 2)}s")

            # saving checkpoint
            if epoch == self.MaxEpoch - 1:
                self.saving_checkpoing(epoch, 'final_model.pth', must_save=True)
            else:
                self.saving_checkpoing(epoch, 'latest_model.pth')
        
        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            self.train_loader._finish()
            self.val_loader._finish()
            sys.stdout = old_stdout

        torch.cuda.empty_cache()
        self.logger.info(f"Training Done.")