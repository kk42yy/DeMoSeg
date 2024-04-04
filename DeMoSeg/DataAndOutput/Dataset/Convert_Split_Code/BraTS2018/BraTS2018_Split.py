import os
import pickle
from collections import OrderedDict
osp = os.path.join

def load_pick(file:str, mode:str='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

Txt_path = os.path.split(os.path.abspath(__file__))[0]
savepath = osp(Txt_path, "datasplits.pkl")

########### fold0
val_txt_file = osp(Txt_path, 'val1.txt')
with open(val_txt_file, 'r') as f:
    val_set = f.readlines()
val_set = [i.split('\n')[0] for i in val_set]

tr_txt_file = osp(Txt_path, 'train1.txt')
with open(tr_txt_file, 'r') as f:
    tr_set = f.readlines()
tr_set = [i.split('\n')[0] for i in tr_set]

valkey = sorted(val_set)
trainkey = sorted(tr_set)

new_info = []

#fold_0: 199train + 29val
new_info.append(OrderedDict())
new_info[-1]['train'] = trainkey
new_info[-1]['val'] = valkey
print(len(new_info[0]['train']), len(new_info[0]['val']))

########### fold1
val_txt_file = osp(Txt_path, 'val2.txt')
with open(val_txt_file, 'r') as f:
    val_set = f.readlines()
val_set = [i.split('\n')[0] for i in val_set]

tr_txt_file = osp(Txt_path, 'train2.txt')
with open(tr_txt_file, 'r') as f:
    tr_set = f.readlines()
tr_set = [i.split('\n')[0] for i in tr_set]

valkey = sorted(val_set)
trainkey = sorted(tr_set)

#fold_1: 199train + 29val
new_info.append(OrderedDict())
new_info[-1]['train'] = trainkey
new_info[-1]['val'] = valkey
print(len(new_info[1]['train']), len(new_info[1]['val']))

########### fold2
val_txt_file = osp(Txt_path, 'val3.txt')
with open(val_txt_file, 'r') as f:
    val_set = f.readlines()
val_set = [i.split('\n')[0] for i in val_set]

tr_txt_file = osp(Txt_path, 'train3.txt')
with open(tr_txt_file, 'r') as f:
    tr_set = f.readlines()
tr_set = [i.split('\n')[0] for i in tr_set]

valkey = sorted(val_set)
trainkey = sorted(tr_set)

#fold_2: 199train + 29val
new_info.append(OrderedDict())
new_info[-1]['train'] = trainkey
new_info[-1]['val'] = valkey
print(len(new_info[2]['train']), len(new_info[2]['val']))


print(new_info)
write_pickle(new_info, savepath)