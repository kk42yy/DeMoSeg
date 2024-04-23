import torch
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch import nn
from torch.utils.data import RandomSampler
from torch.cuda.amp import autocast, GradScaler
from monai import data, transforms
from monai.losses.dice import DiceCELoss
from monai.transforms.transform import MapTransform
from training.loss.loss_scheduler import PolyLRScheduler, DeepSupervisionLoss
from training.deepsupervision.deepsupervision_utils import Downsample_Label_for_DeepSupervision
from training.network.Baseline import Baseline
from util.file_utils import *

class BasicTrainer(object):
    def __init__(self, task, fold=0, basepath='') -> None:
        self.task = task # ['2020', '2018', '2015']
        self.fold = fold
        self.basepath = basepath

        self.MaxEpoch = 1000
        self.Iteration_for_trainingepoch = 250
        self.Iteration_for_validationepoch = 10
        self.saving_latest_checkpoint_every = 25
        self.only_saving_checkpoint = True
        self.deep_supervision = True
        self.batch_size=2
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.pin_memory = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 16
        self.cache_rate = 0.2

    def _get_data_path_for_each(self, patientname_list, database):
        res = []
        for pt in patientname_list:
            cur = {}
            cur["image"] = []
            cur["label"] = f"{database}/labelsTr/{pt}.nii.gz"
            for mod in ['_0000', '_0001', '_0002', '_0003']:
                cur["image"].append(f"{database}/imagesTr/{pt}{mod}.nii.gz")
            res.append(cur)
        return res

    def get_datafile_augmentation(self):
        dataset_base_path = join(self.basepath, 'DataAndOutput', 'Dataset')
        split_file_path = join(dataset_base_path, 'Convert_Split_Code', 'BraTS'+self.task, 'datasplits.pkl')
        datafile_base_path = join(dataset_base_path, 'DataFile', 'BraTS'+self.task)
        Tr_list, Val_list = load_pickle(split_file_path)[self.fold]['train'], load_pickle(split_file_path)[self.fold]['val']
        
        self.Tr_list = self._get_data_path_for_each(Tr_list, datafile_base_path)
        self.Val_list = self._get_data_path_for_each(Val_list, datafile_base_path)
        
        self.train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], image_only=False),
                transforms.EnsureChannelFirstd(keys=["label"]),
                transforms.CropForegroundd(
                    keys=["image", "label"], source_key="image", k_divisible=[128,128,128]
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                
                transforms.RandSpatialCropd(
                    keys=["image", "label"], roi_size=[128,128,128], random_size=False
                ),
                transforms.RandRotated(
                    keys=["image", "label"], range_x=(-30, 30), range_y=(30, 30), range_z=(-30, 30), prob=1.0, mode='nearest', padding_mode='border'
                ),
                transforms.RandScaleIntensityd(keys="image", factors=(0.9, 1.1), prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=1.0),
                transforms.RandGaussianNoised(keys="image"),
                transforms.GaussianSmoothd(keys="image", sigma=1.0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                Transforms_Dimension(keys=["image", "label"], transposed=(2,1,0)),
                transforms.ToTensord(keys=["image", "label"], device='cpu')
            ]
        )
        
        self.val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], image_only=False),
                transforms.EnsureChannelFirstd(keys=["label"]),
                transforms.CropForegroundd(
                    keys=["image", "label"], source_key="image", k_divisible=[128,128,128]
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                
                transforms.RandSpatialCropd(
                    keys=["image", "label"], roi_size=[128,128,128], random_size=False
                ),
                Transforms_Dimension(keys=["image", "label"], transposed=(2,1,0)),
                transforms.ToTensord(keys=["image", "label"], device='cpu')
            ]
        )        

    def get_dataloader(self):
        self.train_dataset = data.CacheDataset(
            data=self.Tr_list, transform=self.train_transform, cache_rate=self.cache_rate, 
            num_workers=self.num_workers, progress=True
        )
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=self.Iteration_for_trainingepoch*self.batch_size),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory = self.pin_memory
        )
        self.val_dataset = data.CacheDataset(
            data=self.Val_list, transform=self.val_transform, cache_rate=self.cache_rate, 
            num_workers=self.num_workers, progress=True
        )
        self.val_loader = data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(self.val_dataset, replacement=True, num_samples=self.Iteration_for_validationepoch*self.batch_size),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory = self.pin_memory
        )

    def get_network(self):
        num_classes = {
            '2020': 4,
            '2018': 4,
            '2015': 5
        }
        self.network = Baseline(input_channels=4, num_classes=num_classes[self.task], num_pool=5)
        self.network.inference_apply_nonlin = torch.nn.Softmax(dim=1)
        self.network.training_apply_nonlin = nn.Identity()
        self.seg_down_sampling = Downsample_Label_for_DeepSupervision(
            ds_scales=[[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(np.vstack([[2]*3 for _ in range(self.network.num_pool)]), axis=0))[:-1],
            order=0
        )

    def get_loss(self):
        loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        if self.deep_supervision:
            weights = np.array([1 / (2 ** i) for i in range(self.network.num_pool)])
            weights[-1] = 0
            weights = weights / weights.sum()
            weights = torch.from_numpy(weights).to(self.device)
            self.loss = DeepSupervisionLoss(loss, weights)
        else:
            self.loss = loss
    
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
                    for idx, trans in enumerate(getattr(self, k).transforms):
                        infor[k][f"DA_{idx}"] = {
                            f"{trans.__class__.__name__}": f"{trans.__dict__}"
                        }
            
            infor['device'] = str(self.device)
            infor['gpu_name'] = torch.cuda.get_device_name()
            infor['torch_version'] = torch.__version__
            infor['cudnn_version'] = torch.backends.cudnn.version()
            save_json(infor, file=self.output_base_folder_path+'/debug.json', sort_keys=False)

        self.logger = logger

    def initialize(self):
        self.get_datafile_augmentation()
        self.get_network()
        self.get_loss()
        self.get_optimizer_scheduler()
        self.initialize_logging()

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
            for idx, batch in enumerate(tqdm(self.train_loader)):
                epoch_loss[idx] = self.training_epoch(batch)

            # batch validation iteration
            with torch.no_grad():
                self.network.eval()
                val_loss, val_dice = np.zeros(self.Iteration_for_validationepoch), np.zeros((self.Iteration_for_validationepoch, self.network.num_classes-1))
                for idx, batch in enumerate(tqdm(self.val_loader)):
                    val_loss[idx], val_dice[idx][:] = self.validation_epoch(batch)

            self.logger.info(f"train_loss: {np.round(np.nanmean(epoch_loss), 4)}")
            self.logger.info(f"val_loss: {np.round(np.nanmean(val_loss), 4)}")
            self.logger.info(f"pseudo val_dice: {np.round(np.nanmean(val_dice, axis=0), 4)}, mean: {np.round(np.mean(val_dice), 4)}")
            self.logger.info(f"epoch time: {np.round(time()-epoch_startime, 2)}s")

            # saving checkpoint
            if epoch == self.MaxEpoch - 1:
                self.saving_checkpoing(epoch, 'final_model.pth', must_save=True)
            else:
                self.saving_checkpoing(epoch, 'latest_model.pth')
        
        torch.cuda.empty_cache()
        self.logger.info(f"Training Done.")

    def validation_epoch(self, batch_data):
        img: torch.Tensor = batch_data['image']
        lbl: torch.Tensor = batch_data['label']
        if self.deep_supervision:
            lbl = self.seg_down_sampling(lbl)

        img = img.to(self.device, non_blocking=True)
        if isinstance(lbl, list):
            lbl = [i.to(self.device, non_blocking=True) for i in lbl]
        else:
            lbl = lbl.to(self.device, non_blocking=True)

        with autocast(enabled=True):
            output = self.network(img)
            l = self.loss(output, lbl)

        if self.deep_supervision:
            output, lbl = output[0], lbl[0]

        output = output.argmax(1)
        lbl = lbl[:,0]
        dice_for_foreground = []
        for b in range(self.batch_size):
            dice_for_foreground.append([])
            for organ in range(1, self.network.num_classes):
                dice_for_foreground[-1].append(self.Dice(output[b]==organ, lbl[b]==organ))
        
        return l.detach().cpu().numpy(), tuple(np.nanmean(dice_for_foreground, axis=0))

    def Dice(self, test=None, reference=None, smooth=1e-5):
        """2TP / (2TP + FP + FN)"""
        TP = int(((test != 0) * (reference != 0)).sum())
        FP = int(((test != 0) * (reference == 0)).sum())
        FN = int(((test == 0) * (reference != 0)).sum())
        return 2*TP / (2*TP + FP + FN + smooth)

    def training_epoch(self, batch_data):
        img: torch.Tensor = batch_data['image']
        lbl: torch.Tensor = batch_data['label']
        if self.deep_supervision:
            lbl = self.seg_down_sampling(lbl)

        img = img.to(self.device, non_blocking=True)
        if isinstance(lbl, list):
            lbl = [i.to(self.device, non_blocking=True) for i in lbl]
        else:
            lbl = lbl.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            output = self.network(img)
            l = self.loss(output, lbl)

        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return l.detach().cpu().numpy()

    def saving_checkpoing(self, epoch, file_suffix, must_save=False):
        if must_save or (epoch+1) % self.saving_latest_checkpoint_every == 0:
            if self.only_saving_checkpoint:
                checkpoint = {'state_dict': self.network.state_dict()}
            else:
                checkpoint = {
                    'state_dict': self.network.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict(),
                    'current_epoch': epoch + 1
                }
            torch.save(checkpoint, f"{self.output_base_folder_path}/{file_suffix}")

class Transforms_Dimension(MapTransform):
    def __init__(
        self,
        keys,
        transposed: tuple,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.transposed = transposed

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key in self.key_iterator(d):
            ds = d[key]
            if len(ds.shape) == 4 and len(self.transposed) == 3:
                new_transposed = [0] + [i+1 for i in self.transposed]
            elif len(ds.shape) == 3 and len(self.transposed) == 3:
                new_transposed = self.transposed
            ds = ds.permute(*new_transposed)
            d[key] = ds
        return d
