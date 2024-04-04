from inference.Predict_from_folder import predict_DeMoSeg

if __name__ == "__main__":
    for modality in range(15):
        print(f"*********\nstarting inferring the missing modality situation: {modality}\n*********")
        predict_DeMoSeg(
            model='.../BraTS20_Model.pth',
            input_folder=f'.../missing_imagesTs/imagesTs_{modality}',
            output_folder=f'.../BraTS2020_Inference/missing_{modality}',
            modality=modality
        )