import os
import torch
import numpy as np
import SimpleITK as sitk

from training.network.DeMoSeg import DeMoSeg
from inference.BasePredictor import BasicPredictor

join = os.path.join

class DeMoSeg_Predictor(BasicPredictor):
    def __init__(self, task='2020', modality=14):
        super().__init__(task, modality)
    
    def initialize_network(self):
        self.network = DeMoSeg(
            input_channels=4,
            base_num_features=32,
            num_classes=5 if self.task == '2015' else 4,
            num_pool=5,
            modality=self.modality
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = torch.nn.Softmax(dim=1)

    @staticmethod
    def predict_DeMoSeg(task: str, model: str, input_folder: str, output_folder: str, modality: int = 14):
        
        os.makedirs(output_folder, exist_ok=True)
        
        case_ids = np.unique([i[:-12] for i in sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))])
        output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
        all_files = sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))
        list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                        len(i) == (len(j) + 12)] for j in case_ids]
        
        predictor = DeMoSeg_Predictor(task=task, modality=modality)
        
        return predictor.predict_cases(model, list_of_lists, output_files)
