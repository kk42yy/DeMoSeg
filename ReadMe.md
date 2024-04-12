# Decoupling Feature Representations of Ego and Other Modalities for Incomplete Multi-modal Brain Tumor Segmentation

<div align="center">
<h2>DeMoSeg</h2>
<p align="center">
    <img src="BraTS20_18_15_Results\DeMoSeg.png"/ width=1000> <br />
</p>
</div>

DeMoSeg, i.e., **De**coupling the task of representing the ego and other **Mo**dalities for robust incomplete multi-modal **Seg**mentation, consists of three major components, (i) feature decoupling of self and mutual expression for each modality, (ii) feature compensation based on clinical prior knowledge, and (iii) a U-Net backbone for tumor segmentation. By decoupling features, the learning burden of modality adaptation is reduced. The proposed novel layer named CSSA and the feature compensation strategy named RCR enable cross-guidance among features effectively. Significant improvements in results on multiple BraTS datasets have validated our method. These novel contributions are vital for brain tumor segmentation under missing-modality scenarios.

## CONTENT: 
- (1)[Concrete Results of BraTS2020, BraTS2018, BraTS2015](#1-Concrete_Results)
- (2)[DeMoSeg Codes and Models](#2-DeMoSeg)
    - 2.1 [Enviroment and Dataset](#21-environment-and-dataset)
    - 2.2 [DeMoSeg Training](#22-DeMoSeg-Training)
    - 2.3 [DeMoSeg Inference](#23-demoseg-inference)
    - 2.4 [Evaluation](#24-evaluation)
- (3)[DeMoSeg_Slicer](#3-demoseg_slicer)

## 1 Concrete_Results 

### BraTS2020
#### BraTS2020 contains 369 training cases which are split into 219, 50 and 100 subjects for training, validation and test, respectively.

| Method                      | WT | TC | ET | 
| --------------------------- | -------- | -------- | -------- |
| RFNet                       |  86.98   |  78.23   |  61.47   |
| mmFormer                    |  86.49   |  76.06   |  63.19   | 
| MAVP                        |  87.58   |  79.67   |  64.87   |
| GSS                         |  88.09   |  79.24   |  66.42   | 
| DeMoSeg                     |  **88.90**   |  **81.58**   |  **69.71**   | 

The results with **bold** represent the best performance, and with <u>underline</u> denote the second best performance.

<p align="left">
    <img src="BraTS20_18_15_Results\BraTS20.png"/ width=1000> <br />
</p>

### BraTS2018
#### BraTS2018 contains 285 training cases which are split into 199, 29 and 57 subjects for training, validation and test, respectively. We use a **three-fold validation** with the same split lists as previous works.

| Method                      | WT | TC | ET | 
| --------------------------- | -------- | -------- | -------- |
| RFNet                       |  85.67   |  76.53   |  57.12   |
| mmFormer                    |  86.38   |  75.82   |  59.12   | 
| MAVP                        |  86.60   |  76.00   |  60.01   |
| GSS                         |  87.20   |  78.25   |  63.49   | 
| DeMoSeg                     |  **87.96**   |  **79.79**   |  **64.90**   | 

<p align="left">
    <img src="BraTS20_18_15_Results\BraTS18.png"/ width=1000> <br />
</p>

### BraTS2015
#### BraTS2015 contains 274 training cases which are split into 242, 12 and 20 subjects for training, validation and test, respectively.

| Method                      | WT | TC | ET | 
| --------------------------- | -------- | -------- | -------- |
| RFNet                       |  86.13   |  71.93   |  64.13   |
| mmFormer                    |  85.67   |  69.96   |  61.77   | 
| MAVP                        |  86.56   |  71.94   |  62.06   |
| GSS                         |  87.59   |  74.16   |  **66.92**   | 
| DeMoSeg                     |  **88.96**   |  **75.88**   |  66.17   | 

<p align="left">
    <img src="BraTS20_18_15_Results\BraTS15.png"/ width=1000> <br />
</p>

## 2 DeMoSeg

We use three BraTS datasets that can be download at [BraTS](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571), or [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?select=BraTS2020_ValidationData). Data splits are consistent with [RFNet](https://github.com/dyh127/RFNet/tree/main).
Here we firstly release the inference code and models trained on BraTS2020.

### 2.1 Environment and Dataset

1) Enviroment: 

    We only support `pytorch==1.12.1, torchvision==0.13.1, torchaudio==0.12.1, cudatoolkit=11.6`, please prepare the enviroment as following.

    ```bash
    conda create -n DeMoSeg python=3.8
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
    cd .../DeMoSeg
    pip install -r requirements.txt
    ```

2) Dataset:

    - Downloading original BraTS2020, BraTS2018 and BraTS2015.
    - Turn images format into NIFTI, using [BraTS20XX_DataConvert.py](DeMoSeg/DataAndOutput/Dataset/Convert_Split_Code/BraTS2020/BraTS2020_DataConvert.py), the [Original Path](DeMoSeg/DataAndOutput/Dataset/Convert_Split_Code/BraTS2020/BraTS2020_DataConvert.py/#L101) should be set firstly.

### 2.2 DeMoSeg Training

1) Dataset split as previous works, like `RFNet`:

    Obtaining the dataset split file by [BraTS20XX_Split.py](DeMoSeg/DataAndOutput/Dataset/Convert_Split_Code/BraTS2020/BraTS2020_Split.py)

2) Training:

    - Set [BraTS Task](DeMoSeg/Train_Baseline.py/#L6), e.g. '2020', '2018' or '2015' and [fold](DeMoSeg/Train_Baseline.py/#L7).
    - Train Baseline or DeMoSeg:
    ```bash
    cd .../DeMoSeg
    python Train_Baseline.py # Baseline
    python Train_DeMoSeg.py  # DeMoSeg
    ```

### 2.3 DeMoSeg Inference

1) Prepare the inference dataset. Taking BraTS20 as an example, 100 testing cases should transformed into `nii.gz` format, and the suffix should obey following correspondence:
    ```python
    correspondence: dict = {
        "T1"   : "_0000.nii.gz",
        "T1ce" : "_0001.nii.gz",
        "T2"   : "_0002.nii.gz",
        "FLAIR": "_0003.nii.gz"
    }
    ```

    Then, to better imitate the missing modality scenarios, and avoiding the impact from preprocessing, we mask the corresponding modality for each missing situation. You can use our provided [Nii_Mask.py](DeMoSeg/util/Nii_Mask.py), the transformed testing images may be as following:
    
        BraTS20/missing_imagesTs
            ├── imagesTs_0
            │   ├── BraTS20_Training_001_0000.nii.gz # T1
            │   ├── BraTS20_Training_001_0001.nii.gz # T1ce , masked by 0 for missing situation 0
            │   ├── BraTS20_Training_001_0002.nii.gz # T2   , masked by 0 for missing situation 0
            │   ├── BraTS20_Training_001_0003.nii.gz # FLAIR, masked by 0 for missing situation 0
            │   ├── BraTS20_Training_016_0000.nii.gz
            │   ├── BraTS20_Training_016_0001.nii.gz
            │   ├── BraTS20_Training_016_0002.nii.gz
            │   ├── BraTS20_Training_016_0003.nii.gz
            |   ...
            ├── imagesTs_1
            |   ...
            ├── imagesTs_13
            └── imagesTs_14


2) Download the trained model [DeMoSeg_BraTS2020](https://drive.google.com/file/d/1WP7A9knH7xW-zI2WiYgodAkzjrun-svY/view?usp=drive_link) and put it anywhere.

3) We release the [infer code](DeMoSeg/Infer_DeMoSeg.py), and using `modality` $\in[0,14]$ to specify the missing modality scenario. Parts of inference codes refer to [nnU-Net V1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). Before starting the inference, please develop the [relevant parameters](DeMoSeg/Infer_DeMoSeg.py/#L10) as below.

    ```python
    for modality in range(15):
        print(f"*********\nstarting inferring the missing modality situation: {modality}\n*********")
        DeMoSeg_Predictor.predict_DeMoSeg(
            task='2020', # BraTS: ['2020', '2018', '2015']
            model='.../BraTS20_Model.pth', # model path
            input_folder=f'.../missing_imagesTs/imagesTs_{modality}', # test images folder path
            output_folder=f'.../BraTS2020_Inference/missing_{modality}', # output folder path
            modality=modality # missing modality situation index
        )
    ```

4) Infer: 
    ```bash
    cd .../DeMoSeg
    python Infer_Baseline.py
    python Infer_DeMoSeg.py
    ```

### 2.4 Evaluation

1) Post-processsing and Evaluation

    The post-processing is used following previous works, like RFNet, MAVP and GSS. We have also prepared **DSC, 95%HD, Sensitivity and Specificity** evaluation code for BraTS2020, BraTS2018 and BraTS2015 at [Evaluation](DeMoSeg/evaluation). Post-processsing and Evaluation codes are integrated, please set the inference output folder path and labels folder path at [infer_bathpath](DeMoSeg/evaluation/BraTS20_Eval.py/#L4) to obtain the results for each missing modality scenario. Finally, in order to get comparison statistical results as shown in the table above, please use [Final_result_statistic.py](DeMoSeg/evaluation/Final_result_statistic.py)

    ```bash
    python .../DeMoSeg/evaluation/BraTS20_Eval.py
    python .../DeMoSeg/evaluation/Final_result_statistic.py
    ```

2) 3-fold CV for BraTS2018

    ```bash
    python .../DeMoSeg/evaluation/BraTS18_Eval.py
    python .../DeMoSeg/evaluation/BraTS18_Statistic.py
    ```

3) AttributeError:

    The `numpy` and `medpy` may have conflict in `np.bool`, please change it to `np.bool_`.
    ```python
    AttributeError: module 'numpy' has no attribute 'bool'.
    ```
    
Our FD, CSSA and RCR codes, BraTS2018 and BraTS2015 models are coming soon.

## 3 DeMoSeg_Slicer

<div align="center">
<h2>
<img src="BraTS20_18_15_Results\BrainTumorDeMoSeg.png" width=50; style="vertical-align: middle;"> DeMoSeg_Slicer
</h2>
<p align="center">
    <img src="BraTS20_18_15_Results\DeMoSeg_Slicer.png"/ width=1000> <br />
</p>
</div>

**DeMoSeg_Slicer** is a useful tool built for 3D Slicer based on our DeMoSeg, capable of handling segmentation of gliomas with 15 missing modality scenarios. The relevant plugin codes are also coming soon.
