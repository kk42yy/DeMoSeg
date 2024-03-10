import os
import torch
import numpy as np
import SimpleITK as sitk
from typing import Tuple
join = os.path.join

from multiprocessing import Pool
from inference.Predictor import DeMoSeg_Predictor
from training.preprocess.preprocess_util import preprocess_multithreaded

def save_segmentation_nifti_from_softmax(segmentation_softmax: np.ndarray, out_fname: str, properties_dict: dict):

    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    seg_old_spacing = segmentation_softmax

    seg_old_spacing = seg_old_spacing.argmax(0)
    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)
    print('saving successfully!', os.path.split(out_fname)[-1])

def predict_cases_DeMoSeg(model, list_of_lists, output_filenames, modality=14):

    pool = Pool(2)
    results = []
    torch.cuda.empty_cache()
    Predictor = DeMoSeg_Predictor(modality=modality)
    params = [torch.load(model, map_location=torch.device('cpu'))]

    preprocessing = preprocess_multithreaded(list_of_lists, output_filenames, 6)
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
            Predictor.network.load_state_dict(params[0]['state_dict'])
            softmax = Predictor.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=True, mirror_axes=(0,1,2),
                step_size=0.5, use_gaussian=True, all_in_gpu=True)[1]

            torch.cuda.empty_cache()
            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                              ((softmax, output_filename, dct),)
                                              ))

    _ = [i.get() for i in results]
    
    pool.close()
    pool.join()

def predict_DeMoSeg(model: str, input_folder: str, output_folder: str, modality: int = 14):
    
    os.makedirs(output_folder, exist_ok=True)
    
    case_ids = np.unique([i[:-12] for i in sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))])
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
    
    return predict_cases_DeMoSeg(model, list_of_lists, output_files, modality)