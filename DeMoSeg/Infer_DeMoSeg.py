import torch
from inference.DeMoSeg_Predictor import DeMoSeg_Predictor

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    for modality in range(15):
        print(f"*********\nstarting inferring the missing modality situation: {modality}\n*********")
        DeMoSeg_Predictor.predict_DeMoSeg(
            task='2020',
            model='.../BraTS20_Model.pth',
            input_folder=f'.../missing_imagesTs/imagesTs_{modality}',
            output_folder=f'.../BraTS2020_Inference/missing_{modality}',
            modality=modality
        )