import torch
import numpy as np
from typing import Tuple, List
from torch.cuda.amp import autocast
from util.data_utils import *
from tqdm import tqdm

def predict_3D(network, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                use_gaussian: bool = False, pad_border_mode: str = "constant",
                pad_kwargs: dict = None, all_in_gpu: bool = False,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    
    torch.cuda.empty_cache()

    if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)
    if pad_kwargs is None:
        pad_kwargs = {'constant_values': 0}

    with autocast():
        with torch.no_grad():
            res = internal_predict_3D_3Dconv_tiled(network, x, step_size, do_mirroring, mirror_axes, patch_size,
                                                    regions_class_order, use_gaussian, pad_border_mode,
                                                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                    verbose=verbose)
            
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

def internal_predict_3D_3Dconv_tiled(network: torch.nn.Module, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                    patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                    pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                    verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    if verbose: print("step_size:", step_size)
    if verbose: print("do mirror:", do_mirroring)
    device = next(network.parameters()).device.index

    data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
    data_shape = data.shape
    num_classes = network.num_classes

    steps = compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    if verbose:
        print("data shape:", data_shape)
        print("patch size:", patch_size)
        print("steps (x, y, and z):", steps)
        print("number of tiles:", num_tiles)

    if use_gaussian and num_tiles > 1:
        gaussian_importance_map = get_gaussian(patch_size, sigma_scale=1. / 8)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        if torch.cuda.is_available():
            gaussian_importance_map = gaussian_importance_map.cuda(device, non_blocking=True)

    else:
        gaussian_importance_map = None

    if use_gaussian and num_tiles > 1:
        gaussian_importance_map = gaussian_importance_map.half()
        gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
            gaussian_importance_map != 0].min()
        add_for_nb_of_preds = gaussian_importance_map
    else:
        add_for_nb_of_preds = torch.ones(patch_size, device=device)

    if verbose: print("initializing result array (on GPU)")
    aggregated_results = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half,
                                        device=device)

    if verbose: print("moving data to GPU")
    data = torch.from_numpy(data).cuda(non_blocking=True)

    if verbose: print("initializing result_numsamples (on GPU)")
    aggregated_nb_of_predictions = torch.zeros([num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                device=device)

    with tqdm(total=len(steps[0])*len(steps[1])*len(steps[2]), colour='blue') as pbar:
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

                    predicted_patch = internal_maybe_mirror_and_pred_3D(network,
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map, device=device)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
        
                    pbar.update(1)

    # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
    slicer = tuple(
        [slice(0, aggregated_results.shape[i]) for i in
            range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

    # computing the class_probabilities by dividing the aggregated result with result_numsamples
    aggregated_results /= aggregated_nb_of_predictions
    del aggregated_nb_of_predictions

    predicted_segmentation = aggregated_results.argmax(0)

    if all_in_gpu:
        if verbose: print("copying results to CPU")

        if regions_class_order is None:
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

        aggregated_results = aggregated_results.detach().cpu().numpy()

    if verbose: print("prediction done")
    return predicted_segmentation, aggregated_results

def internal_maybe_mirror_and_pred_3D(network, x, mirror_axes: tuple,
                                    do_mirroring: bool = True,
                                    mult = None, device=None) -> torch.tensor:
    result_torch = torch.zeros([1, network.num_classes] + list(x.shape[2:]),
                                dtype=torch.float)
    result_torch = result_torch.cuda(device, non_blocking=True)


    if do_mirroring:
        mirror_idx = 8
        num_results = 2 ** len(mirror_axes)
    else:
        mirror_idx = 1
        num_results = 1

    for m in range(mirror_idx):
        if m == 0:
            pred = network.inference_apply_nonlin(network(x))
            result_torch += 1 / num_results * pred

        if m == 1 and (2 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (4, ))))
            result_torch += 1 / num_results * torch.flip(pred, (4,))

        if m == 2 and (1 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (3, ))))
            result_torch += 1 / num_results * torch.flip(pred, (3,))

        if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (4, 3))))
            result_torch += 1 / num_results * torch.flip(pred, (4, 3))

        if m == 4 and (0 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (2, ))))
            result_torch += 1 / num_results * torch.flip(pred, (2,))

        if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (4, 2))))
            result_torch += 1 / num_results * torch.flip(pred, (4, 2))

        if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (3, 2))))
            result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
            pred = network.inference_apply_nonlin(network(torch.flip(x, (4, 3, 2))))
            result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

    if mult is not None:
        result_torch[:, :] *= mult

    return result_torch