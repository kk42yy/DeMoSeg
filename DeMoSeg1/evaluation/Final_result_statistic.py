import pandas as pd
import numpy as np
from collections import OrderedDict

mapping = {
    0: 'T1',
    1: 'T1ce',
    2: 'T2',
    3: 'Flair',
    4: 'T1_T1ce',
    5: 'T1_T2',
    6: 'T1_Flair',
    7: 'T1ce_T2',
    8: 'T1ce_Flair',
    9: 'T2_Flair',
    10: 'T1_T1ce_T2',
    11: 'T1_T1ce_Flair',
    12: 'T1_T2_Flair',
    13: 'T1ce_T2_Flair',
    14: 'T1_T1ce_T2_Flair'
}

mapping_RFNet = [2,1,0,3,7,4,6,5,9,8,11,12,13,10,14]

def read_csv(csv_path, miss_idx=0):
    ckd_data = pd.read_csv(csv_path, usecols=[1,2,3,4,5,6,7,8,9])
    ckd_data = ckd_data.iloc[-5]
    res = []
    for k,v in ckd_data.items():
        ckd_data[k] = np.array(v).astype(np.float64)
        res.append(float(np.array(v).astype(np.float64)))
    res = [str(miss_idx), mapping[miss_idx]] + res
    print(res)
    return res

def static_final_csv(RESULT, savepath):
    SAVE = OrderedDict()
    ITEMS = ['WT_DSC','TC_DSC','ET_DSC','WT_HD95','TC_HD95','ET_HD95', 'WT_SENS','TC_SENS','ET_SENS']
    for title in ['INdex', 'Miss'] + ITEMS:
        SAVE[title] = list()

    # case result
    for case_res in RESULT:
        SAVE['INdex'].append(case_res[0])
        SAVE['Miss'].append(case_res[1])
        for k, dc_or_hd_name in enumerate(ITEMS, start=2):
            dc_or_hd = case_res[k]
            dc_or_hd = round(dc_or_hd, 4) if 'DSC' in dc_or_hd_name else round(dc_or_hd, 3)
            SAVE[dc_or_hd_name].append(dc_or_hd)

    # avg result for case
    for title in ['INdex', 'Miss'] + ITEMS:
        if title == 'INdex':
            SAVE[title].append('AVG')
        elif title == 'Miss':
            SAVE[title].append('')
        else:
            if 'HD95' not in title:
                SAVE[title].append(np.round(np.nanmean(np.array(SAVE[title])), 2))
            else:
                SAVE[title].append(np.round(np.nanmean(np.array(SAVE[title])), 3))

    df = pd.DataFrame(SAVE)
    df.to_csv(savepath, index=False)

if __name__ == "__main__":
    miss_result_bath_dir_path = '.../missing_modality'
    post = True
    RFNet_Order = True
    
    savepath = miss_result_bath_dir_path + (f'/Miss_pp_result.csv' if post else '/Miss_result.csv')
    RESULT = []
    for i in range(15):
        case_csv_result_path = miss_result_bath_dir_path + (f'/missing_{i}_post' if post else f'/missing_{i}') + '/Eval_Sta.csv'
        RESULT.append(read_csv(case_csv_result_path, i))
    static_final_csv(RESULT, savepath)
    
    if RFNet_Order:
        print()
        rfnet_res = []
        for v in mapping_RFNet:
            rfnet_res.append(RESULT[v])
            print(RESULT[v])
        static_final_csv(rfnet_res, miss_result_bath_dir_path + (f'/Miss_pp_result_RFNet_Order.csv' if post else '/Miss_result_RFNet_Order.csv'))
    