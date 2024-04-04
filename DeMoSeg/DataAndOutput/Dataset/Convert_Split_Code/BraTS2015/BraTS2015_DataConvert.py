import os
import shutil
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
osp = os.path.join

# 1. change mha to nii.gz
def change_mha_to_niigz(orip, newp):
    itk = sitk.ReadImage(orip)
    sitk.WriteImage(itk, newp)

# 2. move
def move_patient(datapath, txt_name, TaskPath, mode='Tr'):
    patient_name = txt_name[3:]
    HGG_or_LGG = f"{txt_name[:2]}G"
    msd_dir_path = f"{datapath}/{HGG_or_LGG}/{patient_name}"
    five_dir_name = os.listdir(msd_dir_path)
    for vsd_dir in five_dir_name:
        if 'T1.' in vsd_dir:
            newp = f"{Task_Path}/images{mode}/{txt_name}_0000.nii.gz"
        elif 'T1c.' in vsd_dir:
            newp = f"{Task_Path}/images{mode}/{txt_name}_0001.nii.gz"
        elif 'T2.' in vsd_dir:
            newp = f"{Task_Path}/images{mode}/{txt_name}_0002.nii.gz"
        elif 'Flair.' in vsd_dir:
            newp = f"{Task_Path}/images{mode}/{txt_name}_0003.nii.gz"
        elif 'OT' in vsd_dir:
            newp = f"{Task_Path}/labels{mode}/{txt_name}.nii.gz"
        else:
            raise TypeError(f"{txt_name} {vsd_dir}")
        change_mha_to_niigz(
            f"{msd_dir_path}/{vsd_dir}/{vsd_dir}.mha",
            newp
        )
    
    print(txt_name)

def move_main(datapath, taskpath):
    txt_path = os.path.split(os.path.abspath(__file__))[0]

    test_txt_path = txt_path + '/test.txt'
    with open(test_txt_path, 'r') as f:
        TEST = f.readlines()

    val_txt_path = txt_path + '/val.txt'
    with open(val_txt_path, 'r') as f:
        VAL = f.readlines()

    tar_txt_path = txt_path + '/train.txt'
    with open(tar_txt_path, 'r') as f:
        TRAIN = f.readlines()

    TEST = [i.split('\n')[0] for i in TEST]
    VAL = [i.split('\n')[0] for i in VAL]
    TRAIN = [i.split('\n')[0] for i in TRAIN]
    
    all_arg = []
    for tr in TRAIN + VAL:
        arg = datapath, tr, taskpath, 'Tr'
        all_arg.append(arg)

    for ts in TEST:
        arg = datapath, ts, taskpath, 'Ts'
        all_arg.append(arg)

    
    pool = Pool()
    pool.starmap(move_patient, all_arg)
    pool.close()
    pool.join()

# 3. Copy Info
def copy_each(taskpath, case, mode='Tr'):
    # copy to T1 no matter how different of other
    t1p = f"{taskpath}/images{mode}/{case}_0000.nii.gz"
    t1itk = sitk.ReadImage(t1p)

    for dealing in [
        f"{taskpath}/images{mode}/{case}_0001.nii.gz",
        f"{taskpath}/images{mode}/{case}_0002.nii.gz",
        f"{taskpath}/images{mode}/{case}_0003.nii.gz",
        f"{taskpath}/labels{mode}/{case}.nii.gz"
    ]:
        itk = sitk.ReadImage(dealing)
        itk.CopyInformation(t1itk)
        sitk.WriteImage(itk, dealing)

    print(case)

def copyinfo_main(taskpath):
    Tr = os.listdir(f"{taskpath}/labelsTr")
    Ts = os.listdir(f"{taskpath}/labelsTs")
    
    all_arg = []
    for case in Tr:
        arg = taskpath, case.split('.nii')[0], 'Tr'
        all_arg.append(arg)

    for case in Ts:
        arg = taskpath, case.split('.nii')[0], 'Ts'
        all_arg.append(arg)

    pool = Pool()
    pool.starmap(copy_each, all_arg)
    pool.close()
    pool.join()


if __name__ == "__main__":

    OriDataPath = ''
    Task_Path = '.../DataAndOutput/Dataset/DataFile/BraTS2015'

    MAKEDIRS = True
    MOVE_NII = True
    COPY_INFO = True
    
    if MAKEDIRS:
        _ = [os.makedirs(osp(Task_Path, i), exist_ok=True) for i in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']]

    if MOVE_NII: move_main(OriDataPath, Task_Path)

    if COPY_INFO: copyinfo_main(Task_Path)