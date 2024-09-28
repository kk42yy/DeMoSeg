import os
import json
from collections import OrderedDict
osp = os.path.join

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


savepath = ".../nnUNetV2Frame/DATASET/nnUNet_preprocessed/Dataset010_BraTS20/splits_final.json"
Txt_path = "" # set the split txt path

val_txt_file = osp(Txt_path, 'val.txt')
with open(val_txt_file, 'r') as f:
    val_set = f.readlines()
val_set = [i.split('\n')[0][3:] for i in val_set]

tr_txt_file = osp(Txt_path, 'train.txt')
with open(tr_txt_file, 'r') as f:
    tr_set = f.readlines()
tr_set = [i.split('\n')[0][3:] for i in tr_set]

valkey = sorted(val_set)
trainkey = sorted(tr_set)

new_info = []

#fold_0: 219train + 50val
new_info.append(OrderedDict())
new_info[-1]['train'] = trainkey
new_info[-1]['val'] = valkey
print(len(new_info[0]['train']), len(new_info[0]['val']))

print(new_info)
save_json(new_info, savepath)