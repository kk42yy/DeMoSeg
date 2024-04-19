import os
import torch
import numpy as np
import SimpleITK as sitk

from multiprocessing import Pool
from training.network.Baseline import Baseline
from training.network.sliding_window import predict_3D
from training.preprocess.preprocess_util import preprocess_multithreaded

join = os.path.join

class BasicPredictor(object):
    def __init__(self, task='2020', modality=14):
        super().__init__()
        self.task = task
        self.modality = modality
    
    def initialize_network(self):
        self.network = Baseline(
            input_channels=4,
            base_num_features=32,
            num_classes=5 if self.task == '2015' else 4,
            num_pool=5,
            modality=self.modality
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = torch.nn.Softmax(dim=1)

    def predict_preprocessed_data(self, data, do_mirroring = True, step_size = 0.5):
        
        self.network.do_ds = False
        current_mode = self.network.training
        self.network.eval()
        ret = predict_3D(self.network, data, do_mirroring=do_mirroring, 
                         step_size=step_size,
                         patch_size=[128, 128, 128], 
                         pad_kwargs={'constant_values': 0})
        self.network.train(current_mode)
        
        return ret

    def save_results(self, segmentation_softmax: np.ndarray, out_fname: str, properties_dict: dict):

        shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
        seg_old_spacing = segmentation_softmax.argmax(0)

        bbox = properties_dict.get('crop_bbox')

        if bbox is not None:
            seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
            for c in range(3):
                bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
            seg_old_size[bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]] = seg_old_spacing
        else:
            seg_old_size = seg_old_spacing

        seg_old_size_postprocessed = seg_old_size

        seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, out_fname)
        print('saving successfully!', os.path.split(out_fname)[-1])

    def predict_cases(self, model, list_of_lists, output_filenames):

        pool = Pool(2)
        results = []
        torch.cuda.empty_cache()
        params = [torch.load(model, map_location=torch.device('cpu'))]
        self.initialize_network()
        preprocessing = preprocess_multithreaded(list_of_lists, output_filenames, 6)
        all_output_files = []
        with torch.no_grad():
            for preprocessed in preprocessing:
                output_filename, (d, dct) = preprocessed
                all_output_files.append(all_output_files)
                if isinstance(d, str):
                    data = np.load(d)
                    os.remove(d)
                    d = data

                print("predicting", output_filename)
                self.network.load_state_dict(params[0]['state_dict'])
                softmax = self.predict_preprocessed_data(d, do_mirroring=True, step_size=0.5)[1]

                torch.cuda.empty_cache()
                results.append(pool.starmap_async(self.save_results, ((softmax, output_filename, dct),)))


        _ = [i.get() for i in results]
        
        pool.close()
        pool.join()

    @staticmethod
    def predict(task: str, model: str, input_folder: str, output_folder: str, modality: int = 14):
        
        os.makedirs(output_folder, exist_ok=True)
        
        case_ids = np.unique([i[:-12] for i in sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))])
        output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
        all_files = sorted(j for j in os.listdir(input_folder) if j.endswith('.nii.gz'))
        list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                        len(i) == (len(j) + 12)] for j in case_ids]
        
        predictor = BasicPredictor(task=task, modality=modality)
        
        return predictor.predict_cases(model, list_of_lists, output_files)
