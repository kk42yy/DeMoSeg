import numpy as np
import SimpleITK as sitk
from multiprocessing import Process, Queue
from typing import OrderedDict
from training.preprocess.cropping_resampling import crop, resample_and_normalize

def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties

def preprocess_test_case(data_files, target_spacing=[1,1,1], seg_file=None):
    data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
    data, seg, properties = crop(data, properties, seg)
    data, seg, properties = resample_and_normalize(data, target_spacing, properties, seg,
                                                        force_separate_z=None)
    return data.astype(np.float32), seg, properties

def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files):
    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            d, _, dct = preprocess_fn(l)
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(list_of_lists, output_files, num_processes=2):
    num_processes = min(len(list_of_lists), num_processes)

    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(preprocess_test_case, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes]))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        q.close()