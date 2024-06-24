import torch
import numpy as np
from skimage.transform import resize

class Downsample_Label_for_DeepSupervision():
    def __init__(
        self,
        ds_scales=(1, 0.5, 0.25), order=0, 
        axes=None,
    ) -> None:
        self.axes = axes
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        to_device = False
        if isinstance(data, torch.Tensor):
            device = data.device
            data = data.cpu().numpy()
            to_device = True
        
        res = self.downsample_seg_for_ds_transform2(data, self.ds_scales, self.order, self.axes)
        if to_device:
            res = [torch.from_numpy(i).to(device) for i in res]
        return res

    def downsample_seg_for_ds_transform2(self, seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, axes=None):
        if axes is None:
            axes = list(range(2, len(seg.shape)))
        output = []
        for s in ds_scales:
            if all([i == 1 for i in s]):
                output.append(seg)
            else:
                new_shape = np.array(seg.shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=seg.dtype)
                for b in range(seg.shape[0]):
                    for c in range(seg.shape[1]):
                        out_seg[b, c] = self.resize_segmentation(seg[b, c], new_shape[2:], order)
                output.append(out_seg)
        return output

    def resize_segmentation(self, segmentation, new_shape, order=3):
        tpe = segmentation.dtype
        unique_labels = np.unique(segmentation)
        assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
        if order == 0:
            return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
        else:
            reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

            for i, c in enumerate(unique_labels):
                mask = segmentation == c
                reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
                reshaped[reshaped_multihot >= 0.5] = c
            return reshaped