import os
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

index_missing_dict = {
        0: [1,0,0,0],
        1: [0,1,0,0],
        2: [0,0,1,0],
        3: [0,0,0,1],
        4: [1,1,0,0],
        5: [1,0,1,0],
        6: [1,0,0,1],
        7: [0,1,1,0],
        8: [0,1,0,1],
        9: [0,0,1,1],
        10: [1,1,1,0],
        11: [1,1,0,1],
        12: [1,0,1,1],
        13: [0,1,1,1],
        14: [1,1,1,1]
    }

if __name__ == "__main__":
    # generate 15 distinct missing modalities situation to imitate clinical scenario, rather just dropping features

    orip = '.../Test'
    referp = '.../TestLabel'
    newp_base = '.../missing_imagesTs'
    ts_set = sorted(i[:-7] for i in os.listdir(referp) if i.endswith('.gz'))
    for i in range(15):
        newp = f'{newp_base}/imagesTs_{i}'
        os.makedirs(newp, exist_ok=True)
        t = tqdm(ts_set)
        for pat in t:
            for mod, v_mod in zip(['_0000.nii.gz', '_0001.nii.gz', '_0002.nii.gz', '_0003.nii.gz'], index_missing_dict[i]):
                if v_mod == 1:
                    shutil.copy(orip+'/'+pat+mod, newp+'/'+pat+mod)
                elif v_mod == 0:
                    itk = sitk.ReadImage(orip+'/'+pat+mod)
                    arr = sitk.GetArrayFromImage(itk)
                    arr *= v_mod
                    assert np.unique(arr) == 0
                    itknew = sitk.GetImageFromArray(arr)
                    itknew.CopyInformation(itk)
                    sitk.WriteImage(itknew, newp+'/'+pat+mod)
                else:
                    raise ValueError
            t.set_description(f'mod {i}')
            t.write(pat)