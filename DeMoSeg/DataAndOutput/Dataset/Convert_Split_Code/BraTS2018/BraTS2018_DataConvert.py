import os
import shutil
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
osp = os.path.join

# 1. change nii to nii.gz
def change_nii_to_gz(orip, newp, remove=False):
    itk = sitk.ReadImage(orip)
    sitk.WriteImage(itk, newp)
    if remove: 
        os.remove(orip)

def change_main(datapath, remove=False):
    Dataset_name = sorted([i for i in os.listdir(datapath) if os.path.isdir(osp(datapath, i))])
    all_arg = []
    for patient in Dataset_name:
        nii_path = osp(datapath, patient)
        for mri_or_seg in sorted(i for i in os.listdir(nii_path) if i.endswith('.nii')):
            arg = osp(nii_path, mri_or_seg), osp(nii_path, mri_or_seg+'.gz'), remove
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
            osp(TaskPath, 'images'+mode, patient_name+ind+'.nii.gz')
            )
    shutil.copy(
            osp(mri_seg_dirpath, patient_name+'_seg.nii.gz'),
            osp(TaskPath, 'labels'+mode, patient_name+'.nii.gz')
        )
    print(patient_name)

def move_all_main(datapath, taskpath):
    HGG_Name = sorted(os.listdir(f"{datapath}/HGG"))
    LGG_Name = sorted(os.listdir(f"{datapath}/LGG"))
    all_arg = []
    for hgg in HGG_Name:
        arg = osp(datapath, 'HGG', hgg), taskpath
        all_arg.append(arg)

    for lgg in LGG_Name:
        arg = osp(datapath, 'LGG', lgg), taskpath
        all_arg.append(arg)

    pool = Pool()
    pool.starmap(move_patient, all_arg)
    pool.close()
    pool.join()

def move_testpatient(taskpath, tsname, mode='Tr'):
    mods = ['_0000', '_0001', '_0002', '_0003']
    shutil.copy(
        f"{taskpath}/labelsTr/{tsname}.nii.gz",
        f"{taskpath}/labels{mode}/{tsname}.nii.gz"
    )
    for mod in mods:
        shutil.copy(
            f"{taskpath}/imagesTr/{tsname}{mod}.nii.gz",
            f"{taskpath}/images{mode}/{tsname}{mod}.nii.gz"
        )
    print(tsname)

def move_main(datapath, txtpath, mode='Tr1'):
    test_txt_path = txtpath
    with open(test_txt_path, 'r') as f:
        TEST = f.readlines()

    TEST = [i.split('\n')[0] for i in TEST]

    all_arg = []
    for ts in TEST:
        arg = datapath, ts, mode
        all_arg.append(arg)
    
    pool = Pool()
    pool.starmap(move_testpatient, all_arg)
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

def converse_main(labelpath):
    all_arg = []
    
    for tr in sorted([i for i in os.listdir(labelpath) if i.endswith('.nii.gz')]):
        arg = osp(labelpath, tr),
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
    MOVE_NII_1 = True
    MOVE_NII_2 = True

    """
    20240130
    BraTS2018 has three splits for evaluation:
    1) for imagesTr, 285 cases are stored in
    2) imagesTs, imagesTs1, imagesTs2 store each split test cases
    3) use three different folds for training
    """

    if MAKEDIRS:
        _ = [os.makedirs(osp(Task_Path, i), exist_ok=True) for i in ['imagesTr', 'imagesTs', 'labelsTr', 'imagesTs1', 'labelsTs1', 'imagesTs2', 'labelsTs2', 'imagesTs3', 'labelsTs3']]

    # 1.change nii to nii.gz
    if CHANGE_NII: 
        change_main(f"{OriDataPath}/HGG", True)
        change_main(f"{OriDataPath}/LGG", True)

    # 2.move nii
    
    if MOVE_NII_1: 
        # move all to imagesTr and labelsTr
        move_all_main(OriDataPath, Task_Path)
        # change label_id4 to 3
        converse_main(f"{Task_Path}/labelsTr")

    if MOVE_NII_2: 
        # move splits imagesTs and labelsTs to dir
        basedir = os.path.split(os.path.abspath(__file__))[0]
        move_main(Task_Path, f'{basedir}/test1.txt', 'Ts1')
        move_main(Task_Path, f'{basedir}/test2.txt', 'Ts2')
        move_main(Task_Path, f'{basedir}/test3.txt', 'Ts3')
