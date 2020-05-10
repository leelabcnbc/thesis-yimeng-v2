"""load data from

https://gin.g-node.org/doi/Cadena_PlosCB19_data

Data for the PlosCB 2019 paper Deep convolutional models improve predictions of macaque V1 responses to natural images

https://www.biorxiv.org/content/10.1101/201764v2
"""

from os.path import join
import pickle
import numpy as np
from skimage.transform import downscale_local_mean
from ... import dir_dict
from . import one_shuffle_general

global_dict = {
    'pkl_file': join(dir_dict['private_data'],
                     'Cadena_PlosCB19_data',
                     'data_binned_responses',
                     'cadena_ploscb_data.pkl'
                     ),
    'num_img': 7250,
    'num_neurons_total': 166,
    # neurons with responses to
    'num_neurons_with_response_to_all_images': 115,
    'num_trial': 4,
}


def get_raw_pkl():
    with open(global_dict['pkl_file'], 'rb') as f:
        return pickle.load(f)


def images(px_kept=80, final_size=40):
    x_all = get_raw_pkl()['images']
    assert x_all.shape == (global_dict['num_img'], 140, 140)
    assert px_kept == 80
    assert final_size == 40
    slice_to_use = slice(70 - px_kept // 2, 70 + px_kept // 2)
    x_all = x_all[:, slice_to_use, slice_to_use]
    downscale_ratio = px_kept // final_size
    assert downscale_ratio * final_size == px_kept
    scale_factors = (1, downscale_ratio, downscale_ratio)
    x_all = downscale_local_mean(x_all, scale_factors)[:, np.newaxis]
    assert x_all.shape == (global_dict['num_img'], 1, final_size, final_size)
    assert x_all.min() >= 0
    assert x_all.max() <= 255
    return x_all


def get_neural_data(
        *,
        unit_mean_per_neuron=True,
        post_scale=None,
):
    y = get_neural_data_per_trial(transpose_idx=(0, 1, 2))
    assert y.shape == (
        global_dict['num_trial'],
        global_dict['num_img'],
        global_dict['num_neurons_with_response_to_all_images']
    )
    y = np.nanmean(y, axis=0)
    assert y.shape == (
        global_dict['num_img'],
        global_dict['num_neurons_with_response_to_all_images']
    )
    assert np.all(np.isfinite(y))

    if unit_mean_per_neuron:
        # this is the same type of processing done in 8k data
        per_neuron_mean = y.mean(axis=0)
        assert np.all(per_neuron_mean > 0)
        y = y / per_neuron_mean

    if post_scale is not None:
        y = y * post_scale

    return y


def get_neural_data_per_trial(
        fill_value=None,
        remove_neurons_with_any_nan_mean_resp=True,
        transpose_idx=(2, 0, 1)):
    response_all = get_raw_pkl()['responses']
    assert response_all.shape == (global_dict['num_trial'], global_dict['num_img'], global_dict['num_neurons_total'])
    assert remove_neurons_with_any_nan_mean_resp

    if remove_neurons_with_any_nan_mean_resp:
        good_index = np.logical_not(np.any(np.all(np.isnan(response_all), axis=0), axis=0))
        assert good_index.sum() == global_dict['num_neurons_with_response_to_all_images']
        response_all = response_all[:, :, good_index]

    # fill in
    if fill_value is None:
        # do nothing, keep NaN
        pass
    elif fill_value == 'zero':
        # fill with zero
        response_all[np.isnan(response_all)] = 0
    elif fill_value == 'avg-over-valid-trials':
        # fill with average over valid trials
        avg_over_trials = np.nanmean(response_all, axis=0, keepdims=True)
        assert avg_over_trials.shape == (1, global_dict['num_img'], global_dict['num_neurons_with_response_to_all_images'])
        response_all = np.where(np.isnan(response_all), avg_over_trials, response_all)
    else:
        raise NotImplementedError

    if fill_value is not None:
        assert np.all(np.isfinite(response_all))

    # num_trial x num_image x num_neuron.
    # response_all.transpose((2, 0, 1)) to get num_neuron x num_trial x num_image
    # this is for cc_max computation.
    response_all = response_all.transpose(transpose_idx)

    return response_all


def get_indices(*,
                seed,
                group_by='image_numbers',
                return_dict=True,
                ):
    map_dict = {
        'original': 0,
        'conv1': 1,
        'conv2': 2,
        'conv3': 3,
        'conv4': 4,
    }
    # original, conv1, conv2, conv3, conv4
    labels = [x.tolist()[0] for x in get_raw_pkl()['image_types'].ravel()]
    labels = np.array([map_dict[x] for x in labels])
    assert np.array_equal(np.bincount(labels),
                          np.full((5,), fill_value=global_dict['num_img'] // 5, dtype=np.int64))

    if group_by == 'image_numbers':
        groups = get_raw_pkl()['image_numbers'].ravel()
        assert np.array_equal(groups, np.repeat(np.arange(1, global_dict['num_img'] // 5 + 1), 5))
        train_val_idx, test_idx = one_shuffle_general(
            labels=groups,
            test_size=0.2,
            seed=seed,
            split_type='GroupShuffleSplit'
        )

        train_idx, val_idx = one_shuffle_general(
            labels=groups[train_val_idx],
            test_size=0.2,
            seed=seed,
            split_type='GroupShuffleSplit'
        )

        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]
    else:
        raise NotImplementedError

    np.array_equal(
        np.sort(np.concatenate((train_idx, val_idx, test_idx))),
        np.arange(global_dict['num_img']))

    assert train_idx.size == global_dict['num_img'] // 5 * 4 // 5 * 4
    assert test_idx.size == global_dict['num_img'] // 5
    assert val_idx.size == global_dict['num_img'] // 5 * 4 // 5

    assert np.array_equal(
        np.bincount(labels[train_idx]),
        np.full((5,), fill_value=train_idx.size // 5, dtype=np.int64)
    )

    assert np.array_equal(
        np.bincount(labels[test_idx]),
        np.full((5,), fill_value=test_idx.size // 5, dtype=np.int64)
    )

    assert np.array_equal(
        np.bincount(labels[val_idx]),
        np.full((5,), fill_value=val_idx.size // 5, dtype=np.int64)
    )

    indices = {
        'labels': labels,
        'groups': groups,
        'idx_train': train_idx,
        'idx_val': val_idx,
        'idx_test': test_idx,
    }
    if return_dict:
        return indices
    else:
        return (indices['idx_train'], indices['idx_val'], indices['idx_test'])


def get_data(*, px_kept, final_size,
             seed,
             scale=None,
             ):
    x_all = images(px_kept, final_size)
    assert x_all.shape == (global_dict['num_img'], 1, final_size, final_size)

    y = get_neural_data(post_scale=scale)

    indices = get_indices(seed=seed, return_dict=False)

    result = []
    for idx in indices:
        result.append(x_all[idx])
        if isinstance(y, np.ndarray):
            result.append(y[idx])
        elif isinstance(y, tuple):
            result.append(tuple(y_this[idx] for y_this in y))
        else:
            raise NotImplementedError

    return tuple(result)
