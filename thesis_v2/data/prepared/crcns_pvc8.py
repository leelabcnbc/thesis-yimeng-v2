"""module to get prepared CRCNS data set

we will use a caching mechanism to avoid preparing data explicitly.

this makes coding easier.
"""
from functools import partial
import numpy as np
from skimage.transform import downscale_local_mean

from . import (join, dir_root,
               split_stuff, one_shuffle_general)
from ..raw import load_data
from .. import load_data_lazy_helper


def natural_data_neural(*, window_size, read_only=True, scale=None):
    fname_y = join(dir_root, 'crcns_pvc-8_neural.hdf5')
    key_y = f'{window_size}/center/natural'
    func_y = partial(process_crcns_pvc8_neural_data,
                     window_size=window_size,
                     natural_only=True,
                     centered=True
                     )

    y = load_data_lazy_helper(key_y, func_y, fname=fname_y,
                              read_only=read_only)

    # for changing scales (for data-driven CNN, etc.)

    if scale is not None:
        y = y * scale

    return y


def natural_data(window_size, px_kept, downscale_ratio,
                 seed, scale=None, trans_x=None,
                 read_only=True, shuffle_type='legacy'):
    # previous way to prepared this data.
    # key thing is to generate some key along the way
    # so things can be cached.
    fname_x = join(dir_root, 'crcns_pvc-8_images.hdf5')
    assert type(window_size) is str
    assert type(px_kept) is int
    assert type(downscale_ratio) is int
    key_x = f'{window_size}/keep{px_kept}/down{downscale_ratio}/natural'
    func_x = partial(process_crcns_pvc8_image,
                     window_size=window_size,
                     px_kept=px_kept,
                     downscale_ratio=downscale_ratio,
                     natural_only=True
                     )

    x_all = load_data_lazy_helper(key_x, func_x, fname=fname_x,
                                  read_only=read_only)

    y = natural_data_neural(window_size=window_size, read_only=read_only, scale=scale)

    # then you can apply some custom transformer to adapt the data
    # for different settings.

    if trans_x is not None:
        trans_x_mu, trans_x_std = trans_x
        x_all = (x_all - trans_x_mu) / trans_x_std

    # time for splitting.
    num_im = 540
    assert x_all.shape[0] == y.shape[0] == num_im
    # the preprocessing one (train + val) is ignored.
    if shuffle_type == 'legacy':
        idx_sets = get_idx_sets_legacy(num_im, seed)[:3]
    elif shuffle_type == 'paired':
        idx_sets = get_idx_sets_natural_grouped(seed, return_dict=False)[:3]
    else:
        raise NotImplementedError

    return split_stuff(x_all, y, idx_sets)


# from <https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal/data_aux/neural_datasets.py>  # noqa: E501
def process_crcns_pvc8_image(window_size, px_kept, downscale_ratio,
                             natural_only):
    assert window_size in {'large', 'medium'}
    # load X
    x_all = load_data('crcns_pvc-8_images', window_size)
    assert x_all.shape == (956, 320, 320)

    if natural_only:
        image_idx_slice = slice(540)
    else:
        raise NotImplementedError

    # then crop images.
    assert 0 < px_kept <= 320
    assert px_kept % downscale_ratio == 0
    slice_to_use = slice(160 - px_kept // 2, 160 + px_kept // 2)
    x_all = x_all[image_idx_slice, slice_to_use, slice_to_use]
    scale_factors = (1, downscale_ratio, downscale_ratio)
    x_all = downscale_local_mean(x_all, scale_factors)[:, np.newaxis]

    # I will leave all the preprocessing later.

    return x_all


def process_crcns_pvc8_neural_data(window_size, natural_only, centered, mean=True):
    # load y.
    if window_size == 'large':
        sections = range(1, 8)
    elif window_size == 'medium':
        sections = range(8, 11)
    else:
        raise ValueError

    if natural_only:
        image_idx_slice = slice(540)
    else:
        raise NotImplementedError

    y_all = []

    if mean:
        field = 'mean'
    else:
        field = 'all_shifted'

    # then load data from all sizes
    for section in sections:
        y_this = load_data('crcns_pvc-8_neural',
                           '{:02d}/{}'.format(section, field))

        if centered:
            good_idx_this = load_data('crcns_pvc-8_neural',
                                      '{:02d}/attrs/INDCENT'.format(section))
        else:
            raise NotImplementedError
        if mean:
            y_this = y_this[image_idx_slice, good_idx_this]
        else:
            y_this = y_this[image_idx_slice, :, good_idx_this]
        y_all.append(y_this)
        # print('section', section, y_this.shape)

    y_all = np.concatenate(y_all, axis=-1)
    return y_all


def _one_shuffle(groups, test_size, seed):
    return one_shuffle_general(labels=groups, test_size=test_size, seed=seed,
                               split_type='GroupShuffleSplit')


def get_idx_sets_natural_grouped(seed,
                                 return_dict=True,
                                 by='pairs'
                                 ):
    # in this way, small and large images don't appear together.
    if by == 'pairs':
        groups = np.arange(270).repeat(2)
    else:
        raise NotImplementedError
    assert groups.shape == (540,)
    train_val_idx, test_idx = _one_shuffle(groups, 0.2, seed)
    train_idx, val_idx = _one_shuffle(groups[train_val_idx], 0.2, seed)
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    np.array_equal(
        np.sort(np.concatenate((train_idx, val_idx, test_idx))),
        np.arange(540))
    np.array_equal(
        np.sort(np.concatenate((train_val_idx, test_idx))),
        np.arange(540))
    # print(train_idx.shape, test_idx.shape, val_idx.shape)
    assert train_idx.size == 344
    assert test_idx.size == 108
    assert val_idx.size == 88
    if return_dict:
        return {
            'groups': groups,
            'idx_train': train_idx,
            'idx_val': val_idx,
            'idx_test': test_idx,
            'idx_train_val': train_val_idx,
        }
    else:
        # compatible with get_idx_sets
        return train_idx, val_idx, test_idx, train_val_idx


# from <https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal/preprocessing.py>  # noqa: E501
def _sanity_check_valid_idx(idx_list, n):
    idx_all = np.concatenate(idx_list)
    assert np.array_equal(np.sort(idx_all), np.arange(n))


def get_idx_sets_legacy(num_im, seed):
    # ok. time to get indices.
    rng_state = np.random.RandomState(seed=seed)
    num_train = int(num_im * 0.8 * 0.8)
    num_val = int(num_im * 0.8) - num_train
    num_test = num_im - num_train - num_val
    print('train', num_train, 'val', num_val, 'test', num_test)
    assert num_train > 0 and num_val > 0 and num_test > 0
    perm = rng_state.permutation(num_im)
    idx_train = perm[:num_train]
    idx_val = perm[num_train:num_train + num_val]
    idx_test = perm[-num_test:]

    idx_preprocessing = perm[:-num_test]

    _sanity_check_valid_idx((idx_train, idx_val, idx_test), num_im)
    _sanity_check_valid_idx((idx_preprocessing, idx_test), num_im)

    return idx_train, idx_val, idx_test, idx_preprocessing
