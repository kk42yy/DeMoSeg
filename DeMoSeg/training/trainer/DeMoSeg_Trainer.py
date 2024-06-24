import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
from time import time
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from util.file_utils import *
from monai import transforms
from monai.losses.dice import DiceCELoss
from training.network.DeMoSeg import DeMoSeg
from training.loss.loss_scheduler import PolyLRScheduler, DeepSupervisionLoss
from training.trainer.BaseTrainer import BasicTrainer, Transforms_Dimension
from training.dataloader.MultiThreadDataLoader import MultiThreadedDataLoader
from training.dataloader.Augmentation import get_training_transforms, get_validation_transforms
from training.deepsupervision.deepsupervision_utils import Downsample_Label_for_DeepSupervision

class DeMoSeg_Trainer(BasicTrainer):
    def __init__(self, task='2020', fold=0, basepath='') -> None:
        super().__init__(task, fold, basepath)
        self.MaxEpoch = 1000
        self.initial_lr = 1e-2
        self.Iteration_for_trainingepoch = 250
        self.Iteration_for_validationepoch = 10
        self.upsample = False

    def get_network(self):
        num_classes = {
            '2020': 4,
            '2018': 4,
            '2015': 5
        }
        self.network = DeMoSeg(input_channels=4, num_classes=num_classes[self.task], num_pool=5, upsample=self.upsample)
        self.network.inference_apply_nonlin = torch.nn.Softmax(dim=1)
        if self.upsample:
            self.network.training_apply_nonlin = torch.nn.Softmax(dim=1)
        else:
            self.network.training_apply_nonlin = nn.Identity()
        self.seg_down_sampling = Downsample_Label_for_DeepSupervision(
            ds_scales=[[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(np.vstack([[2]*3 for _ in range(self.network.num_pool)]), axis=0))[:-1],
            order=0
        )

    def _get_data_path_for_each(self, patientname_list, database):
        res = []
        for pt in patientname_list:
            cur = {}
            cur["data"] = []
            cur["seg"] = f"{database}/labelsTr/{pt}.nii.gz"
            for mod in ['_0000', '_0001', '_0002', '_0003']:
                cur["data"].append(f"{database}/imagesTr/{pt}{mod}.nii.gz")
            res.append(cur)
        return res
    
    def get_datafile_augmentation(self):
        dataset_base_path = join(self.basepath, 'DataAndOutput', 'Dataset')
        split_file_path = join(dataset_base_path, 'Convert_Split_Code', 'BraTS'+self.task, 'datasplits.pkl')
        datafile_base_path = join(dataset_base_path, 'DataFile', 'BraTS'+self.task)
        Tr_list, Val_list = load_pickle(split_file_path)[self.fold]['train'], load_pickle(split_file_path)[self.fold]['val']
        
        self.Tr_list = self._get_data_path_for_each(Tr_list, datafile_base_path)
        self.Val_list = self._get_data_path_for_each(Val_list, datafile_base_path)
        
        self.train_transform = [transforms.Compose(
            [
                transforms.LoadImaged(keys=["data", "seg"], image_only=False),
                transforms.EnsureChannelFirstd(keys=["seg"]),
                transforms.CropForegroundd(
                    keys=["data", "seg"], source_key="data", k_divisible=[128]*3
                ),
                transforms.NormalizeIntensityd(keys="data", nonzero=True, channel_wise=True),
                
                transforms.RandSpatialCropd(
                    keys=["data", "seg"], roi_size=[128,128,128], random_size=False
                ),
                Transforms_Dimension(keys=["data", "seg"], transposed=(2,1,0))
            ]
        ),
        get_training_transforms(use_mask_for_norm=[True]*4)]
        
        self.val_transform = [transforms.Compose(
            [
                transforms.LoadImaged(keys=["data", "seg"], image_only=False),
                transforms.EnsureChannelFirstd(keys=["seg"]),
                transforms.CropForegroundd(
                    keys=["data", "seg"], source_key="data", k_divisible=[128]*3
                ),
                transforms.NormalizeIntensityd(keys="data", nonzero=True, channel_wise=True),
                transforms.RandSpatialCropd(
                    keys=["data", "seg"], roi_size=[128,128,128], random_size=False
                ),
                Transforms_Dimension(keys=["data", "seg"], transposed=(2,1,0)),
            ]
        ), get_validation_transforms()]

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
    

    def get_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.MaxEpoch)
        self.grad_scaler = GradScaler()


    def initialize_logging(self, saving_debug=True):
        self.output_base_folder_path = f"{self.basepath}/DataAndOutput/Output/BraTS{self.task}/{self.network.__class__.__name__}/fold_{self.fold}"
        makedirs(self.output_base_folder_path)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{self.output_base_folder_path}/training.log")
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        if saving_debug:
            infor = {}
            for k in sorted(self.__dir__(), key=str.lower):
                if not k.startswith("__") and not k.endswith('list'):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        infor[k] = str(getattr(self, k))
                
                if k.endswith("transform"):
                    infor[k] = {}
                    infor[k]['monai'] = {}
                    for idx, trans in enumerate(getattr(self, k)[0].transforms):
                        infor[k]['monai'][f"DA_{idx}"] = {
                            f"{trans.__class__.__name__}": f"{trans.__dict__}"
                        }
                    infor[k]['nnunet'] = {}
                    for idx, trans in enumerate(getattr(self, k)[1].transforms):
                        infor[k]['nnunet'][f"DA_{idx}"] = {
                            f"{trans.__class__.__name__}": f"{trans.__dict__}"
                        }
            
            infor['device'] = str(self.device)
            infor['gpu_name'] = torch.cuda.get_device_name()
            infor['torch_version'] = torch.__version__
            infor['cudnn_version'] = torch.backends.cudnn.version()
            save_json(infor, file=self.output_base_folder_path+'/debug.json', sort_keys=False)

        self.logger = logger
        self.logger.info(self.network)

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