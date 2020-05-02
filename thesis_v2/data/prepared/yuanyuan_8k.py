"""module to get prepared CRCNS data set

we will use a caching mechanism to avoid preparing data explicitly.

this makes coding easier.
"""
from functools import partial
import numpy as np
from skimage.transform import downscale_local_mean
from scipy.io import loadmat
import h5py

from . import (join, dir_root, one_shuffle_general)
from ..raw import load_data
from ... import dir_dict
from .. import load_data_lazy_helper
from ...spike_data_processing.yuanyuan_8k import config_8k
from ...spike_data_processing.yuanyuan_8k.io_8k import load_spike_count_and_meta_data


def images(group, px_kept, final_size, read_only=True):
    # previous way to prepared this data.
    # key thing is to generate some key along the way
    # so things can be cached.
    fname_x = join(dir_root, 'yuanyuan_8k_images.hdf5')
    assert type(group) is str and group in {'a', 'b', 'c'}
    assert type(px_kept) is int

    assert type(final_size) is int and final_size > 0
    key_x = f'group{group}/keep{px_kept}/size{final_size}'
    func_x = partial(process_yuanyuan8k_image,
                     group=group,
                     px_kept=px_kept,
                     final_size=final_size
                     )

    x_all = load_data_lazy_helper(key_x, func_x, fname=fname_x,
                                  read_only=read_only)
    return x_all


def labels_dict(group):
    labels = load_data('yuanyuan_8k_images', group + '/names')
    labels = np.char.decode(labels).tolist()
    # should be a (8000,) unicode string array
    labels = np.asarray([n[:n.index('_')] for n in labels])
    classes, labels = np.unique(labels, return_inverse=True)
    assert labels.shape == (8000,) and labels.dtype == np.int64
    classes = classes.tolist()
    assert len(classes) == 161
    for c in classes:
        assert type(c) is str
    return {
        'labels': labels,
        'classes': classes,
    }


# from <https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal/data_aux/neural_datasets.py>  # noqa: E501
def process_yuanyuan8k_image(group, px_kept, final_size):
    # load X
    x_all = load_data('yuanyuan_8k_images', group)['images']
    assert x_all.shape == (8000, 400, 400)
    # then crop images.
    assert px_kept % 2 == 0 and 0 < px_kept <= 400
    # this is because right now we use mean pooling for downsampling.
    downscale_ratio = px_kept // final_size
    assert downscale_ratio * final_size == px_kept
    slice_to_use = slice(200 - px_kept // 2, 200 + px_kept // 2)
    x_all = x_all[:, slice_to_use, slice_to_use]
    scale_factors = (1, downscale_ratio, downscale_ratio)
    x_all = downscale_local_mean(x_all, scale_factors)[:, np.newaxis]

    # I will leave all the preprocessing later.

    return x_all


def _one_shuffle(labels, test_size, seed):
    return one_shuffle_general(labels=labels, test_size=test_size,
                               seed=seed, split_type='StratifiedShuffleSplit')


def get_data_split_labels(group, seed):
    labels = labels_dict(group)['labels']
    train_val_idx, test_idx = _one_shuffle(labels, 0.2, seed)
    train_idx, val_idx = _one_shuffle(labels[train_val_idx], 0.2, seed)
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    np.array_equal(
        np.sort(np.concatenate((train_idx, val_idx, test_idx))),
        np.arange(8000))
    assert train_idx.size == 5120
    assert test_idx.size == 1600
    assert val_idx.size == 1280
    return {
        'labels': labels,
        'idx_train': train_idx,
        'idx_val': val_idx,
        'idx_test': test_idx,
    }


def get_neural_data(date_list, scale=None):
    y = []
    for date in date_list:
        y.append(load_data('yuanyuan_8k_neural', date)['resp'])
    y = np.concatenate(y, axis=1)

    if scale is not None:
        y = y * scale

    return y


def get_neural_data_per_trial(date_list, scale=None, transpose_idx=(1, 0, 2)):
    response_all = []
    with h5py.File(join(config_8k.result_root_dir, 'responses.hdf5'), 'r') as f:
        for d in date_list:
            response_all.append(f[d]['response_all'][()])
    response_all = np.concatenate(response_all, axis=1)
    if scale is not None:
        response_all = response_all * scale

    # num_trial x num_neuron x num_image.
    # response_all.transpose((1, 0, 2)) to get num_neuron x num_trial x num_image
    # this is for cc_max computation.

    response_all = response_all.transpose(transpose_idx)

    return response_all


def get_per_trial_previous_k_images(*, date_list, k):
    assert 0 < k < config_8k.frame_per_image
    # make sure everything is consistent.
    para_file_mapping = config_8k.para_file_mapping_dict[date_list[0]]
    imageset_mapping = config_8k.imageset_mapping_dict[date_list[0]]
    for d in date_list[1:]:
        assert config_8k.para_file_mapping_dict[d] == para_file_mapping
        assert config_8k.imageset_mapping_dict[d] == imageset_mapping

    # return a num_img x num_trial x k result.
    # -2 represents NA
    result = np.full((len(para_file_mapping), 8000, k), fill_value=-2, dtype=np.int64)
    num_condition = 500
    assert config_8k.frame_per_image * num_condition == 8000
    # go over each trial
    record_paras = load_spike_count_and_meta_data(date_list[0], load_spike=False)['record_paras']
    for idx_trial, param_id in enumerate(para_file_mapping):
        # get the order of images in this one.
        img_original_order = record_paras[param_id, 0][0, 1].ravel()
        assert np.array_equal(np.sort(img_original_order), np.arange(1, 8000 + 1))
        img_original_order = img_original_order - 1
        assert np.array_equal(np.sort(img_original_order), np.arange(8000))
        # img_original_position[x] gives the original position of image x in the 8000 images
        img_original_position = np.argsort(img_original_order)
        assert np.array_equal(np.sort(img_original_position), np.arange(8000))

        for img_idx, original_loc in enumerate(img_original_position):
            # say if original_loc is 16 (first image in the second condition), then num_img_before is 0,
            # if original_loc is 20, num_image_before is 4, i.e. 16, 17, 18, 19.
            assert img_original_order[original_loc] == img_idx
            num_img_before = original_loc % config_8k.frame_per_image
            num_img_before_useful = min(k, num_img_before)
            num_no_img = k - num_img_before_useful

            # :0 is fine.
            # fill oldest `num_no_img` frames with -1
            result[idx_trial, img_idx, :num_no_img] = -1

            for k_this in range(num_img_before_useful):
                # print(num_img_before_useful)
                assert original_loc - (k_this + 1) >= 0
                result[idx_trial, img_idx, -(k_this + 1)] = img_original_order[original_loc - (k_this + 1)]

    # everything is filled.
    assert np.all(result != -2)
    return result


def get_indices(group, seed):
    if seed == 'legacy':
        # split data according to Yuanyuan's way
        idx_set = loadmat(
            join(dir_dict['private_data_supp'], 'yuanyuan_8k_idx.mat'))
        indices = tuple(
            np.flatnonzero(idx_set[k].ravel().astype(np.bool_)) for k in (
                'I_train', 'I_valid', 'I_test'
            ))
    else:
        # splitting is done later.
        # split accroding to labels.
        # based on https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/scripts/preprocessing/split_datasets.py#L152-L166  # noqa: E501
        # get all numerical labels
        indices = get_data_split_labels(group, seed)
        indices = (indices['idx_train'], indices['idx_val'],
                   indices['idx_test'])
    return indices


date_compatibility_map = {
    'a': {
        '042318',
        '043018',
        '051018',
    },
    'b': {
        '050718',
        '051118',
    },
    'c': {
        '050918',
    },
}


def flatten_per_trial_data_one(one_element_in_result_tuple):
    return np.concatenate(one_element_in_result_tuple, axis=0)


def get_data_per_trial(group, px_kept, final_size, date_list,
                       *, read_only=True, seed='legacy',
                       scale=None,
                       bg_color=121.0,
                       previous_k_frames=0,
                       ):
    assert 0 <= previous_k_frames < config_8k.frame_per_image
    assert type(group) is str and group in date_compatibility_map.keys()
    assert set(date_list) <= date_compatibility_map[group]

    x_all = images(group, px_kept, final_size, read_only=read_only)
    assert x_all.shape == (8000, 1, final_size, final_size)

    # before applying response_all.transpose(transpose_idx), num_trial x num_neuron x num_image.
    # response_all.transpose((0, 2, 1)) to get num_trial x num_image x num_neuron
    y = get_neural_data_per_trial(date_list, scale, transpose_idx=(0, 2, 1))

    num_trial, num_image, num_neuron = y.shape
    assert num_image == 8000

    # append frames
    if previous_k_frames == 0:
        # naive. just replicate images.
        full_x_all = np.broadcast_to(x_all, (num_trial,) + x_all.shape)
    else:
        previous_trial_idx = get_per_trial_previous_k_images(date_list = date_list, k = previous_k_frames)
        # previous_trial_idx is of shape (num_trial, num_img, k)
        assert previous_trial_idx.shape == (num_trial, num_image, previous_k_frames)
        assert np.all(np.logical_or(previous_trial_idx >= -1, previous_trial_idx < num_image))
        previous_trial_idx[previous_trial_idx == -1] = num_image
        empty_img = np.full(shape=(final_size, final_size), fill_value=bg_color, dtype=x_all.dtype)
        # fill stuffs.
        # use channel dim to create time. this is kind of a convenient hack.
        full_x_all = np.full((num_trial, num_image, previous_k_frames + 1, final_size, final_size),
                             fill_value=np.nan, dtype=x_all.dtype)
        full_x_all[:, :, -1:] = x_all

        x_all_append = np.concatenate([x_all, empty_img[np.newaxis,np.newaxis]], axis=0)
        assert x_all_append.shape == (num_image + 1, 1, final_size, final_size)
        # advanced np indexing.
        # full_x_all[:, :, :-1] has shape (num_trial, num_image, previous_k_frames, final_size, final_size)
        # x_all_append[previous_trial_idx, 0] has
        # shape (num_trial, num_image, previous_k_frames, final_size, final_size)
        full_x_all[:, :, :-1] = x_all_append[previous_trial_idx, 0]

    assert np.all(np.isfinite(full_x_all))

    del x_all
    # then create subsets
    # full_x_all have shape (num_trial, num_image, previous_k_frames + 1, final_size, final_size)
    # get index set
    # then repeat it for each trial.
    indices = get_indices(group, seed)

    result = []
    for idx in indices:
        result.append(full_x_all[:, idx])
        if isinstance(y, np.ndarray):
            result.append(y[:, idx])
        else:
            raise NotImplementedError

    # result is similar to `get_data`, except that now everything has a (num_trial,) dimension in the front.
    return tuple(result)


def get_data(group, px_kept, final_size,
             date_list,
             *, read_only=True, seed='legacy',
             scale=None, load_labels=False,
             ):
    # legacy means the seed used in
    # <https://github.com/leelabcnbc/cnn-model-leelab-8000/blob/7d8e86141c3219bc154b7c57960e85b780f70257/leelab_8000/get_leelab_8000.m>  # noqa: E501
    assert type(group) is str and group in date_compatibility_map.keys()
    assert set(date_list) <= date_compatibility_map[group]

    x_all = images(group, px_kept, final_size, read_only=read_only)
    assert x_all.shape == (8000, 1, final_size, final_size)

    y = get_neural_data(date_list, scale)

    if load_labels:
        y = (y, labels_dict(group)['labels'])

    indices = get_indices(group, seed)

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
