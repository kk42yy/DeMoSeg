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

Three dataset can be download at [BraTS](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571), or [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?select=BraTS2020_ValidationData).

1) We firstly release our [DeMoSeg Model](https://drive.google.com/file/d/1eItnFqxyJcJ5i6-FFCyFgCzqnJEmhCXm/view?usp=drive_link) on BraTS2020.

2) We release the [infer code](Infer.py), and using `modality` $\in[0,14]$ to point the missing modality scenario. Before starting the inference, please develop the [relevant parameters](Infer.py/#L87) as below.

    ```python
    def predict_DeMoSeg(
        model='.../BraTS20_Model.pth', # model path
        input_folder='...', # test images dir path
        output_folder='...', # output dir path
        modality=0 # missing modality situation index
    )
    ```

3) Infer: 
    ```python
    python Infer.py
    ```

Our network and training codes will release later.
