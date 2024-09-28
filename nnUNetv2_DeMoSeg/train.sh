conda activate nnUNet2

export CUDA_VISIBLE_DEVICES=2

nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer_DeMoSeg