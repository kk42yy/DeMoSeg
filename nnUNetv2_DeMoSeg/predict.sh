conda activate nnUNet2

export CUDA_VISIBLE_DEVICES=2

imgTs=".../nnUNetV2Frame/DATASET/nnUNet_raw/Dataset010_BraTS20/missing_imagesTs/imagesTs_"
outputdir=".../nnUNetV2Frame/DATASET/nnUNet_results/Dataset010_BraTS20/nnUNetTrainer_DeMoSeg__nnUNetPlans__3d_fullres/fold_0/missing_modality/missing_"
dataset=10
trainer="nnUNetTrainer_DeMoSeg"
fold=0
config="3d_fullres"
chk="checkpoint_final.pth"

for mod in $(seq 14 -1 0)
do
nnUNetv2_predict -i $imgTs$mod -o $outputdir$mod -c $config -d $dataset -tr $trainer -f $fold -modality $mod -chk $chk
done
