from os.path import join, exists
from os import makedirs
import h5py
import numpy as np
from json import dump
from numpy.linalg import norm


def postprocess_maskcnn_polished_with_rcnn_k_bl(*, rcnn_bl_cls, global_vars, key_script, dataset_name, file_to_save,
                                                max_len=10, rcnn_acc_type):
    assert rcnn_bl_cls <= max_len + 1
    if rcnn_bl_cls > 1:
        diff_stats_dir = join(global_vars['feature_file_dir'], 'diff_stats_' + key_script)
        # compute stats, computing diff
        makedirs(diff_stats_dir, exist_ok=True)
        # find for files with name `SUCCESS_test`
        stat_file_marker = join(diff_stats_dir, 'SUCCESS_' + dataset_name)
        if not exists(stat_file_marker):
            # then compute stats.
            # save per image stats, as well as overall.
            # overall stats
            with h5py.File(file_to_save, 'r') as f:
                g = f['test']

                if rcnn_acc_type != 'last':
                    resp_neurons = [g[f'1.{x}'][()] for x in range(rcnn_bl_cls)]
                    diff_vec_neurons = generate_diff_vec_overall(
                        tensor_list=resp_neurons,
                        acc_mode='instant',
                        ndim=2,
                        max_len=max_len,
                    )
                    del resp_neurons
                else:
                    diff_vec_neurons = None

                resp_map = [g[f'0.{x}'][()] for x in range(rcnn_bl_cls)]
                diff_vec_map_instant = generate_diff_vec_overall(
                    tensor_list=resp_map,
                    acc_mode='instant',
                    ndim=4,
                    max_len=max_len,
                )

                diff_vec_map_cummean = generate_diff_vec_overall(
                    tensor_list=resp_map,
                    acc_mode='cummean',
                    ndim=4,
                    max_len=max_len,
                )

                # per image stats

                diff_vec_map_instant_dict = generate_diff_vec_per_image(
                    tensor_list=resp_map,
                    acc_mode='instant',
                    max_len=max_len,
                )

                diff_vec_map_cummean_dict = generate_diff_vec_per_image(
                    tensor_list=resp_map,
                    acc_mode='cummean',
                    max_len=max_len,
                )

                # write aux file.
            auxfile = join(diff_stats_dir, dataset_name + '.overall.json')
            with open(auxfile, 'wt', encoding='utf-8') as f_aux:
                dump({
                    'diff_final_act': diff_vec_neurons,
                    'diff_bl_stack_instant': diff_vec_map_instant,
                    'diff_bl_stack_cummean': diff_vec_map_cummean,
                },
                    f_aux,
                )
            auxfile2 = join(diff_stats_dir, dataset_name + '.per_image.json')
            with open(auxfile2, 'wt', encoding='utf-8') as f_aux2:
                dump({
                    'diff_bl_stack_instant': diff_vec_map_instant_dict,
                    'diff_bl_stack_cummean': diff_vec_map_cummean_dict,
                },
                    f_aux2,
                )
            # create this file
            with open(stat_file_marker, 'wt'):
                pass
            assert exists(stat_file_marker)


def generate_diff_vec(*, tensor_list, acc_mode, max_len, ndim):
    assert len(tensor_list) > 1

    if acc_mode == 'instant':
        pass
    elif acc_mode == 'cummean':
        tensor_list_new = []
        for i in range(len(tensor_list)):
            tensor_list_new.append(np.mean(np.asarray(tensor_list[:i + 1]), axis=0))
        tensor_list = tensor_list_new
    else:
        raise ValueError

    # tensor_shape = tensor_list[0].shape
    # get norm diff
    diff_vec = np.full((max_len,), fill_value=np.nan, dtype=np.float64)

    for i in range(len(tensor_list) - 1):
        vec_prev = tensor_list[i]
        vec_now = tensor_list[i + 1]
        assert vec_prev.shape == vec_now.shape
        assert vec_prev.ndim == ndim
        diff_this = norm(vec_now.ravel() - vec_prev.ravel()) / norm(vec_prev.ravel())
        diff_vec[i] = diff_this
    return diff_vec


def generate_diff_vec_per_image(*, tensor_list, acc_mode, max_len):
    num_img = tensor_list[0].shape[0]
    # assert num_img == 1600
    diff_vec_list = []
    for idx_img in range(num_img):
        tensor_list_this_img = [x[idx_img] for x in tensor_list]
        diff_vec_list.append(generate_diff_vec(
            tensor_list=tensor_list_this_img,
            acc_mode=acc_mode,
            max_len=max_len,
            ndim=3,
        ))
    diff_vec_list = np.asarray(diff_vec_list)
    assert diff_vec_list.shape == (num_img, max_len)
    diff_vec_mean = diff_vec_list.mean(axis=0)
    diff_vec_std = diff_vec_list.std(axis=0)
    return {
        'mean': diff_vec_mean.tolist(),
        'std': diff_vec_std.tolist(),
        'raw': diff_vec_list.tolist(),
    }


def generate_diff_vec_overall(*, tensor_list, acc_mode, max_len, ndim):
    return {
        'raw': generate_diff_vec(tensor_list=tensor_list, acc_mode=acc_mode,
                                 max_len=max_len, ndim=ndim).tolist(),
    }
