# Decoupling Feature Representations of Ego and Other Modalities for Incomplete Multi-modal Brain Tumor Segmentation

DeMoSeg, i.e., **De**coupling the task of representing the ego and other **Mo**dalities for robust incomplete multi-modal **Seg**mentation, consists of three major components, (i) feature decoupling of self and mutual expression for each modality, (ii) feature compensation based on clinical prior knowledge, and (iii) a U-Net backbone for tumor segmentation. By decoupling features, the learning burden of modality adaptation is reduced. The proposed novel layer named CSSA and the feature compensation strategy named RCR enable cross-guidance among features effectively. Significant improvements in results on multiple BraTS datasets have validated our method. These novel contributions are vital for brain tumor segmentation under missing-modality scenarios.

## CONTENT: 
- (1)[Concrete Results of BraTS2020, BraTS2018, BraTS2015](#1_Concrete_Results)
- (2)[DeMoSeg Codes and Models](#2_DeMoSeg)

## 1_Concrete_Results 

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

<img src="BraTS20_18_15_Results\BraTS20.png" weight="50px" />

### BraTS2018
#### BraTS2018 contains 285 training cases which are split into 199, 29 and 57 subjects for training, validation and test, respectively. We use a **three-fold validation** with the same split lists as previous works.

| Method                      | WT | TC | ET | 
| --------------------------- | -------- | -------- | -------- |
| RFNet                       |  85.67   |  76.53   |  57.12   |
| mmFormer                    |  86.38   |  75.82   |  59.12   | 
| MAVP                        |  86.60   |  76.00   |  60.01   |
| GSS                         |  87.20   |  78.25   |  63.49   | 
| DeMoSeg                     |  **87.96**   |  **79.79**   |  **64.90**   | 

<img src="BraTS20_18_15_Results\BraTS18.png" weight="50px" />

### BraTS2015
#### BraTS2015 contains 274 training cases which are split into 242, 12 and 20 subjects for training, validation and test, respectively.

| Method                      | WT | TC | ET | 
| --------------------------- | -------- | -------- | -------- |
| RFNet                       |  86.13   |  71.93   |  64.13   |
| mmFormer                    |  85.67   |  69.96   |  61.77   | 
| MAVP                        |  86.56   |  71.94   |  62.06   |
| GSS                         |  87.59   |  74.16   |  **66.92**   | 
| DeMoSeg                     |  **88.96**   |  **75.88**   |  66.17   | 

<img src="BraTS20_18_15_Results\BraTS15.png" weight="50px" />

## 2_DeMoSeg

We use three BraTS datasets that can be download at [BraTS](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571), or [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?select=BraTS2020_ValidationData). Data splits are consistent with [RFNet](https://github.com/dyh127/RFNet/tree/main).
Here we firstly release the infer code and models trained on BraTS.

### 2.1 Inference of DeMoSeg

1) Prepare the BraTS Dataset. Taking BraTS20 as an example, 100 testing cases should transformed into `nii.gz` format, and the suffix should obey following correspondence:
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

3) We release the [infer code](DeMoSeg/Infer.py), and using `modality` $\in[0,14]$ to specify the missing modality scenario. Before starting the inference, please develop the [relevant parameters](DeMoSeg/Infer.py/#L6) as below.

    ```python
    def predict_DeMoSeg(
        model='.../BraTS20_Model.pth', # model path
        input_folder='...', # test images folder path
        output_folder='...', # output folder path
        modality=0 # missing modality situation index
    ): 
        pass

    for modality in range(15):
        print(f"*********\nstarting inferring the missing modality situation: {modality}\n*********")
        predict_DeMoSeg(
            model='.../BraTS20_Model.pth',
            input_folder=f".../missing_imagesTs/imagesTs_{modality}",
            output_folder=f'.../BraTS2020_Inference/missing_{modality}',
            modality=modality
        )
    ```

4) Infer: 
    ```bash
    python .../DeMoSeg/Infer.py
    ```

5) Post-processsing and Evaluation. The post-processing is used following previous works, like RFNet, MAVP and GSS. We have also prepared **DSC, 95%HD, Sensitivity and Specificity** evaluation code for BraTS2020, BraTS2018 and BraTS2015 at [Evaluation](DeMoSeg/evaluation). Post-processsing and Evaluation codes are integrated, please set the inference output folder path and labels folder path at [infer_bathpath](DeMoSeg/evaluation/BraTS20_Eval.py/#L4) to obtain the results for each missing modality scenario. Finally, in order to get comparison statistical results as shown in the table above, please use [Final_result_statistic.py](DeMoSeg/evaluation/Final_result_statistic.py)

    ```bash
    python .../DeMoSeg/evaluation/BraTS20_Eval.py
    python .../DeMoSeg/evaluation/Final_result_statistic.py
    ```
    

Our FD, CSSA and RCR codes, training codes and BraTS2018, BraTS2015 models are comming soon.
