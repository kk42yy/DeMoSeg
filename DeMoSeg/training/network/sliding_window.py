import torch
import itertools
import numpy as np
from typing import Tuple, List
from torch.cuda.amp import autocast
from util.data_utils import *
from tqdm import tqdm

def predict_3D(network, x: np.ndarray, do_mirroring: bool = True, 
                step_size: float = 0.5, patch_size: Tuple[int, ...] = None, 
                pad_kwargs: dict = None, ) -> Tuple[np.ndarray, np.ndarray]:
    
    torch.cuda.empty_cache()

    if pad_kwargs is None:
        pad_kwargs = {'constant_values': 0}

    with autocast():
        with torch.no_grad():
            res = predict_3D_tiled(network, x, step_size, do_mirroring, 
                                   patch_size, pad_kwargs=pad_kwargs)
            
    return res

def compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

def predict_3D_tiled(network: torch.nn.Module, x: np.ndarray, 
                     step_size: float, do_mirroring: bool, 
                     patch_size: tuple, pad_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
    device = next(network.parameters()).device.index

    data, slicer = pad_nd_image(x, patch_size, "constant", pad_kwargs, True, None)
    data_shape = data.shape
    num_classes = network.num_classes

    steps = compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    if num_tiles > 1:
        gaussian_importance_map = get_gaussian(patch_size, sigma_scale=1. / 8)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        if torch.cuda.is_available():
            gaussian_importance_map = gaussian_importance_map.cuda(device, non_blocking=True)

    else:
        gaussian_importance_map = None

    if num_tiles > 1:
        gaussian_importance_map = gaussian_importance_map.half()
        gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
            gaussian_importance_map != 0].min()
        add_for_nb_of_preds = gaussian_importance_map
    else:
        add_for_nb_of_preds = torch.ones(patch_size, device=device)

    aggregated_results = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half, device=device)
    data = torch.from_numpy(data).cuda(non_blocking=True)
    aggregated_nb_of_predictions = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half, device=device)

    with tqdm(total=num_tiles, colour='blue') as pbar:
        pbar.set_description("mirroring TTA")
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = mirror_and_pred_3D(network,
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], do_mirroring,
                        gaussian_importance_map, device=device)[0]

                    predicted_patch = predicted_patch.half()
                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
        
                    pbar.update(1)

    slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

    aggregated_results /= aggregated_nb_of_predictions
    del aggregated_nb_of_predictions

    predicted_segmentation = aggregated_results.argmax(0)

    aggregated_results = aggregated_results.detach().cpu().numpy()
    return predicted_segmentation, aggregated_results

def mirror_and_pred_3D(network, x, 
                       do_mirroring: bool = True,
                       mult = None, device=None) -> torch.tensor:
    pred = network.inference_apply_nonlin(network(x))

    if do_mirroring:
        mirror_axes = (0, 1, 2)
        mirror_axes_iter = [c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)]
        for axes in mirror_axes_iter:
            pred += torch.flip(network.inference_apply_nonlin(network(torch.flip(x, (*axes,)))), (*axes,))
        pred /= (len(mirror_axes_iter) + 1)

    if mult is not None:
        pred[:, :] *= mult

    return pred