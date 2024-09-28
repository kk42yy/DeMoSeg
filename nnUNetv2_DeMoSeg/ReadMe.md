### 2024.09.28 DeMoSeg with nnU-Net V2

1. Follow the [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet/tree/master) official pipline to install `nnUNet V2`
2. Replace or add files as below:
	
	(1) Create dir and place `FD_CSSA_RCR_2random.py` at : 
	```bash
	.../nnUNetV2Frame/nnUNet/nnunetv2/selfnetwork/DeMoSeg/FD_CSSA_RCR_2random.py
	```
	(2) Add `nnUNetTrainer_DeMoSeg.py` at : 
	```bash
	.../nnUNetV2Frame/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_DeMoSeg.py
	```
	(3) Replace `predict_from_raw_data.py` at : 
	```bash
	.../nnUNetV2Frame/nnUNet/nnunetv2/inference/predict_from_raw_data.py
	```

3. Train and Infer can use **train.sh** or **predict.sh**, we set the index of BraTS2020 as Dataset010.


### Example：

1. Dataset Preparation：
```
	nnUNetV2Frame/DATASET/nnUNet_raw/Dataset010_BraTS20
		├──imagesTr
		│   ├── BraTS20_Training_002_0000.nii.gz
		│   ├── BraTS20_Training_002_0001.nii.gz
		│   ├── BraTS20_Training_002_0002.nii.gz
		│   ├── BraTS20_Training_002_0003.nii.gz
		|   ...
		├──imagesTs
		│   ├── BraTS20_Training_001_0000.nii.gz
		│   ├── BraTS20_Training_001_0001.nii.gz
		│   ├── BraTS20_Training_001_0002.nii.gz
		│   ├── BraTS20_Training_001_0003.nii.gz
		|   ...
		├──labelsTr
		│   ├── BraTS20_Training_002.nii.gz
		|   ...
		├──labelsTs
		│   ├── BraTS20_Training_001.nii.gz
		| ...
		├──missing_imagesTs
		│	├── imagesTs_0
		│	│   ├── BraTS20_Training_001_0000.nii.gz # T1
		│	│   ├── BraTS20_Training_001_0001.nii.gz # T1ce , masked by 0 for missing situation 0
		│	│   ├── BraTS20_Training_001_0002.nii.gz # T2   , masked by 0 for missing situation 0
		│	│   ├── BraTS20_Training_001_0003.nii.gz # FLAIR, masked by 0 for missing situation 0
		│	│   ├── BraTS20_Training_016_0000.nii.gz
		│	│   ├── BraTS20_Training_016_0001.nii.gz
		│	│   ├── BraTS20_Training_016_0002.nii.gz
		│	│   ├── BraTS20_Training_016_0003.nii.gz
		│	|   ...
		│	├── imagesTs_1
		│	|   ...
		│	├── imagesTs_13
		│	└── imagesTs_14
		│
		└──dataset.json
```

 preprocess:
```
nnUNetv2_plan_and_preprocess -d 10
```

2. train
```
source train.sh
```

3. infer
```
source predict.sh
```