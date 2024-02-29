# Decoupling Feature Representations of Ego and Other Modalities for Incomplete Multi-modal Brain Tumor Segmentation

## Concrete results of BraTS2020, BraTS2018 and BraTS2015

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
