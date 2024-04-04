import os
import numpy as np
import SimpleITK as sitk

def read_nii_to_arr(nii_path: str) -> np.ndarray:
    itk = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(itk)
    return arr, itk

def write_arr_to_nii(arr: np.ndarray, ref: sitk.Image, savepath: str):
    itk = sitk.GetImageFromArray(arr)
    itk.CopyInformation(ref)
    sitk.WriteImage(itk, savepath)
    return itk

def converse_brats_4class_to_3class(itk_path):
    arr, itk = read_nii_to_arr(itk_path)
    arr[arr == 4] = 3
    assert 4 not in np.unique(arr)

    write_arr_to_nii(arr, itk, itk_path)
    print(os.path.split(itk_path)[-1])

def change_mha_to_niigz(orip, newp):
    itk = sitk.ReadImage(orip)
    sitk.WriteImage(itk, newp)
    
def move_patient(datapath, txt_name, Task_Path, mode='Tr'):
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