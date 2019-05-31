"""module to get prepared CRCNS data set

we will use a caching mechanism to avoid preparing data explicitly.

this makes coding easier.
"""
from functools import partial
import numpy as np
from skimage.transform import downscale_local_mean
from scipy.io import loadmat

from . import (join, dir_root, one_shuffle_general)
from ..raw import load_data
from ... import dir_dict
from .. import load_data_lazy_helper


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


def get_data(group, px_kept, final_size,
             date_list,
             *, read_only=True, seed='legacy',
             scale=None, load_labels=False,
             ):
    # legacy means the seed used in
    # <https://github.com/leelabcnbc/cnn-model-leelab-8000/blob/7d8e86141c3219bc154b7c57960e85b780f70257/leelab_8000/get_leelab_8000.m>  # noqa: E501

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
