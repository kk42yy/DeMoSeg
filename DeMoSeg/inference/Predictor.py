import torch
from torch.nn import functional as F
from training.network.DeMoSeg import DeMoSeg
from training.network.sliding_window import predict_3D

class DeMoSeg_Predictor(object):
    def __init__(self, modality=14):
        super().__init__()
        self.modality = modality
        self.initialize()
    
    def initialize(self):
        self.network = DeMoSeg(
            input_channels=4,
            base_num_features=32,
            num_classes=4,
            num_pool=5,
            modality = self.modality
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = lambda x: F.softmax(x, 1)

    def predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring = True,
                                                         mirror_axes = None,
                                                         step_size = 0.5,
                                                         use_gaussian = True, pad_border_mode = 'constant',
                                                         pad_kwargs = None, all_in_gpu = False,
                                                         verbose = False):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        self.network.do_ds = False
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}
        if do_mirroring and mirror_axes is None:
            mirror_axes = (0, 1, 2)
        current_mode = self.network.training
        self.network.eval()
        ret = predict_3D(self.network, data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                         step_size=step_size,
                         patch_size=[128, 128, 128], regions_class_order=None,
                         use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                         pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose)
        self.network.train(current_mode)
        return ret