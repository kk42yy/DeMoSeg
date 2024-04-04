import os
import shutil
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
osp = os.path.join

# 1. change nii to nii.gz
def change_nii_to_gz(orip, newp):
    itk = sitk.ReadImage(orip)
    sitk.WriteImage(itk, newp)
    os.remove(orip)

def change_main(datapath):
    Dataset_name = sorted([i for i in os.listdir(datapath) if os.path.isdir(osp(datapath, i))])
    all_arg = []
    for patient in Dataset_name:
        nii_path = osp(datapath, patient)
        for mri_or_seg in sorted(i for i in os.listdir(nii_path) if i.endswith('.nii')):
            arg = osp(nii_path, mri_or_seg), osp(nii_path, mri_or_seg+'.gz')
            all_arg.append(arg)
    
    pool = Pool()
    pool.starmap(change_nii_to_gz, all_arg)
    pool.close()
    pool.join()

# 2. move nii
def move_patient(mri_seg_dirpath, TaskPath, mode='tr'):
    patient_name = os.path.split(mri_seg_dirpath)[-1]
    for mod, ind in zip(['_t1', '_t1ce', '_t2', '_flair'], ['_0000', '_0001', '_0002', '_0003']):
        shutil.copy(
            osp(mri_seg_dirpath, patient_name+mod+'.nii.gz'),
            osp(TaskPath, 'imagesTr' if mode=='tr' else 'imagesTs', patient_name+ind+'.nii.gz')
            )
    shutil.copy(
            osp(mri_seg_dirpath, patient_name+'_seg.nii.gz'),
            osp(TaskPath, 'labelsTr' if mode=='tr' else 'labelsTs', patient_name+'.nii.gz')
        )
    print(patient_name)

def move_main(datapath, Task_path):
    Dataset_name = sorted([i for i in os.listdir(datapath) if os.path.isdir(osp(datapath, i))])
    test_txt_path = os.path.split(os.path.abspath(__file__))[0] + '/test.txt'
    with open(test_txt_path, 'r') as f:
        TEST = f.readlines()

    TEST = [i.split('\n')[0][3:] for i in TEST]
    TRAIN = [i for i in Dataset_name if i not in TEST]
    
    all_arg = []
    for tr in TRAIN:
        arg = osp(datapath, tr), Task_path, 'tr'
        all_arg.append(arg)

    for ts in TEST:
        arg = osp(datapath, ts), Task_path, 'ts'
        all_arg.append(arg)

    
    pool = Pool()
    pool.starmap(move_patient, all_arg)
    pool.close()
    pool.join()

# 3. converse
def converse_patient(itk_path):
    itk = sitk.ReadImage(itk_path)
    arr = sitk.GetArrayFromImage(itk)
    arr[arr == 4] = 3
    uniq_arr = np.unique(arr)

    assert 4 not in uniq_arr

    new_itk = sitk.GetImageFromArray(arr)
    new_itk.CopyInformation(itk)
    sitk.WriteImage(new_itk, itk_path)

    print(os.path.split(itk_path)[-1])

def converse_main(taskpath):
    label_tr_p = osp(taskpath, 'labelsTr')
    label_ts_p = osp(taskpath, 'labelsTs')
    all_arg = []
    
    for tr in sorted([i for i in os.listdir(label_tr_p) if i.endswith('.nii.gz')]):
        arg = osp(label_tr_p, tr),
        all_arg.append(arg)

    for ts in sorted([i for i in os.listdir(label_ts_p) if i.endswith('.nii.gz')]):
        arg = osp(label_ts_p, ts),
        all_arg.append(arg)

    pool = Pool()
    pool.starmap(converse_patient, all_arg)
    pool.close()
    pool.join()

if __name__ == "__main__":

    OriDataPath = ''
    Task_Path = '.../DataAndOutput/Dataset/DataFile/BraTS2020'

    MAKEDIRS = True
    CHANGE_NII = False
    MOVE_NII = True
    CONVERSE = True

    if MAKEDIRS:
        _ = [os.makedirs(osp(Task_Path, i), exist_ok=True) for i in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']]

    # 1.change nii to nii.gz
    if CHANGE_NII: change_main(OriDataPath)

    # 2.move nii
    if MOVE_NII: move_main(OriDataPath, Task_Path)

    # 3.change label_id4 to 3
    if CONVERSE: converse_main(Task_Path)