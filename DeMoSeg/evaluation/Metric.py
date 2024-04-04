# author: ... Tue 17th, Oct, 2023 20:26:30
# evaluation for Brain Tumor Segmentation
"""
Label:
    0: background
    1: NCR, the necrotic tumor core
    2: ED, the peritumoral edematous/invaded tissue
    4: ET, Gd-enhancing tumor

Region:
    whole tumor region <WT>: label 1 + 2 + 4
    tumor core region <TC>: label 1 + 4
    enhancing tumor tumor <ET>: label 4
"""

import os
import shutil
import pandas as pd
import numpy as np
import SimpleITK as sitk
from medpy import metric
from multiprocessing import Pool
from typing import OrderedDict

def postprocess_MSD_each(orip, newp, T, ET=3):
    itk = sitk.ReadImage(orip)
    arr = sitk.GetArrayFromImage(itk)
    if (ETs := np.sum(arr==ET)) < T:
        arr[arr == ET] = 1
        new_itk = sitk.GetImageFromArray(arr)
        new_itk.CopyInformation(itk)
        sitk.WriteImage(new_itk, newp)
        print(os.path.split(newp)[-1], f'finishing postprocess with the voxels = {ETs}')
    else:
        shutil.copy(orip, newp)
        print(os.path.split(newp)[-1], 'no changing')

def postprocess_MSD_main(infer_path, post_path, voxel_threshold=100, ET=3):
    assert infer_path != post_path
    os.makedirs(post_path, exist_ok=True)
    all_arg = []
    for i in sorted(i for i in os.listdir(infer_path) if i.endswith('.nii.gz')):
        arg = os.path.join(infer_path, i), os.path.join(post_path, i), voxel_threshold, ET
        all_arg.append(arg)
    pool = Pool()
    pool.starmap(postprocess_MSD_each, all_arg)
    pool.close()
    pool.join()

def Dice(test=None, reference=None, smooth=0.):
    """2TP / (2TP + FP + FN)"""
    if (np.sum(test) < 1. and np.sum(reference) > 0.) or (np.sum(test) > 0. and np.sum(reference) < 1.):
        return 0.
    if np.sum(test) < 1. and np.sum(reference) < 1.:
        return 1.
    
    TP = int(((test != 0) * (reference != 0)).sum())
    FP = int(((test != 0) * (reference == 0)).sum())
    FN = int(((test == 0) * (reference != 0)).sum())
    return 2*TP / (2*TP + FP + FN + smooth)

def hausdorff_distance_95(test=None, reference=None, voxel_spacing=None, nan_for_nonexisting=True, connectivity=1):

    test_empty = not np.any(test)
    test_full = np.all(test)
    reference_empty = not np.any(reference)
    reference_full = np.all(reference)
    
    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    return np.round(metric.hd95(test, reference, voxel_spacing, connectivity), 4)

def sensentivity(test, reference):

    if np.sum(test) < 1. and np.sum(reference) < 1.:
        fp = fn = 0
        tp = tn = 1
        sens = spec = 1
        return [sens, spec]
    
    if np.sum(reference) < 1.:
        sens = tp = fn = 0
        tn = np.sum(np.logical_and(np.logical_not(test), np.logical_not(reference)))
        fp = np.sum(np.logical_and(test, np.logical_not(reference)))
        spec = tn / (tn + fp)
        return [sens, spec]
    
    tp = np.sum(np.logical_and(test, reference))
    tn = np.sum(np.logical_and(np.logical_not(test), np.logical_not(reference)))
    fp = np.sum(np.logical_and(test, np.logical_not(reference)))
    fn = np.sum(np.logical_and(np.logical_not(test), reference))

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    return [sens, spec]

def calculate_DSC_HD_Sen(arr_infer, arr_label, region_list, dc_func, hd_func, sen_func):
    DSC, HD, SEN_SPEC = [], [], []
    res = []
    for region in region_list:
        test=sum(arr_infer == i for i in region)
        reference=sum(arr_label == i for i in region)
        DSC.append(dc_func(test,reference))
        HD.append(hd_func(test, reference))
        SEN_SPEC.append(sen_func(test, reference))
    
    del test, reference

    SEN, SPEC = [], []
    for sen, spec in SEN_SPEC:
        SEN.append(sen)
        SPEC.append(spec)
    
    for i in DSC+HD+SEN+SPEC:
        res.append(i)
    
    return res

def eval_each_case(infer_case_path, 
                   label_case_path, 
                   region_list = [
                       [1,2,3],
                       [1,3],
                       [3]
                   ]):

    case_name = os.path.split(infer_case_path)[-1].split('.nii')[0]
    res = [case_name]
    
    itk_infer = sitk.ReadImage(infer_case_path)
    itk_label = sitk.ReadImage(label_case_path)
    arr_infer = sitk.GetArrayFromImage(itk_infer) #[z,y,x]
    arr_label = sitk.GetArrayFromImage(itk_label)

    infer_spacing = itk_infer.GetSpacing()[::-1] # [z,y,x]
    label_spacing = itk_label.GetSpacing()[::-1]
    assert np.all(np.array(infer_spacing) == np.array(label_spacing)), \
        f"infer and label spacing are not the same, {infer_spacing} and {label_spacing}"

    res.extend(
        calculate_DSC_HD_Sen(
            arr_infer=arr_infer,
            arr_label=arr_label,
            region_list=region_list,
            dc_func=Dice,
            hd_func=hausdorff_distance_95,
            sen_func=sensentivity
        )
    )
    
    print(res)
    return res

def eval_main(infer_dirpath: str, label_dirpath: str, region_list: list):
    infer_set = sorted([i for i in os.listdir(infer_dirpath) if i.endswith('.nii.gz')])
    label_set = sorted([i for i in os.listdir(label_dirpath) if i.endswith('.nii.gz')])
    all_arg = []

    for infer, label in zip(infer_set, label_set):
        assert infer == label
        arg = os.path.join(infer_dirpath, infer), os.path.join(label_dirpath, label), region_list
        all_arg.append(arg)

    pool = Pool()
    RESULT = pool.starmap(eval_each_case, all_arg)
    pool.close()
    pool.join()

    SAVE = OrderedDict()
    ITEMS = ['WT_DSC','TC_DSC','ET_DSC','WT_HD95','TC_HD95','ET_HD95','WT_SENS','TC_SENS','ET_SENS','WT_SPEC','TC_SPEC','ET_SPEC']
    for title in ['Name'] + ITEMS:
        SAVE[title] = list()

    # case result
    for case_res in RESULT:
        SAVE['Name'].append(case_res[0])
        for k, dc_or_hd_name in enumerate(ITEMS, start=1):
            dc_or_hd = case_res[k]
            dc_or_hd = round(dc_or_hd*100, 4) if 'HD95' not in dc_or_hd_name else round(dc_or_hd, 3)
            SAVE[dc_or_hd_name].append(dc_or_hd)
    
    # avg result for case
    for title in ['Name'] + ITEMS:
        if title == 'Name':
            SAVE[title].append('AVG')
        else:
            if 'HD95' not in title:
                SAVE[title].append(np.round(np.nanmean(np.array(SAVE[title])), 2))
            else:
                SAVE[title].append(np.round(np.nanmean(np.array(SAVE[title])), 3))

    # avg totally
    for cnt, avg_str in enumerate(['AVG_DSC', 'AVG_HD95', 'AVG_SENS', 'AVG_SPEC']):
        SAVE['Name'].append(avg_str)
        if 'HD95' not in avg_str:
            SAVE['WT_DSC'].append(np.round(np.nanmean(np.array([SAVE[i][-cnt-1] for i in ITEMS[cnt*3:cnt*3+3]])), 2))
        else:
            SAVE['WT_DSC'].append(np.round(np.nanmean(np.array([SAVE[i][-cnt-1] for i in ITEMS[cnt*3:cnt*3+3]])), 3))
        for j in ITEMS[1:]:
            SAVE[j].append('     ')

    df = pd.DataFrame(SAVE)
    df.to_csv(os.path.join(infer_dirpath, 'Eval_Sta.csv'), index=False)