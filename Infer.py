import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from multiprocessing import Pool
from inference import *

import DeMoSeg
join = os.path.join


def predict_cases_DeMoSeg(model, list_of_lists, output_filenames, modality=14):

    pool = Pool(2)
    results = []
    torch.cuda.empty_cache()
    trainer = DeMoSeg_Trainer('', 0, '', '', False, 0, True, False, True, modality=modality)
    trainer.process_plans(torch.load(model)['plans'])
    trainer.output_folder = os.path.split(model)[0]
    trainer.output_folder_base = os.path.split(model)[0]
    trainer.update_fold(0)
    trainer.initialize(False)
    params = [torch.load(model, map_location=torch.device('cpu'))]

    preprocessing = preprocess_multithreaded(trainer, list_of_lists, output_filenames, 6, None)
    all_output_files = []
    with torch.no_grad():
        for preprocessed in preprocessing:
            output_filename, (d, dct) = preprocessed
            all_output_files.append(all_output_files)
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            print("predicting", output_filename)
            trainer.network.load_state_dict(params[0]['state_dict'])
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=True, mirror_axes=(0,1,2), use_sliding_window=True,
                step_size=0.5, use_gaussian=True, all_in_gpu=True,
                mixed_precision=True)[1]

            torch.cuda.empty_cache()
            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                              ((softmax, output_filename, dct, 1, None,
                                                None, None,
                                                None, None, False, 1),)
                                              ))

    _ = [i.get() for i in results]
    
    pool.close()
    pool.join()

class DeMoSeg_Trainer(BaseTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, modality=14):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.modality = modality
    
    def initialize(self, train):
        self.setup_DA_params()
        self.network = DeMoSeg(
            input_channels=4,
            num_classes=4,
            modality = self.modality
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = lambda x: F.softmax(x, 1)

def predict_DeMoSeg(model: str, input_folder: str, output_folder: str, modality: int = 14):
    
    os.makedirs(output_folder, exist_ok=True)
    
    case_ids = np.unique([i[:-12] for i in sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))])
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
    
    return predict_cases_DeMoSeg(model, list_of_lists, output_files, modality)

if __name__ == "__main__":
    predict_DeMoSeg(
        model='.../BraTS20_Model.pth',
        input_folder='',
        output_folder='',
        modality=0
    )
