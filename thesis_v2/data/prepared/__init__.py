# write a generic stuff to cache data

from os import makedirs
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from ... import dir_dict, abspath, join

dir_root = abspath(join(dir_dict['datasets'], 'prepared'))
makedirs(dir_root, exist_ok=True)


def one_shuffle_general(*,
                        labels, test_size, seed,
                        split_type,
                        ):
    if split_type == 'StratifiedShuffleSplit':
        shuffler = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                          random_state=seed)
        y = labels
        groups = None
    elif split_type == 'GroupShuffleSplit':
        shuffler = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
        y = None
        groups = labels
    else:
        raise NotImplementedError

    counter = 0
    # to suppress code style warnings.
    train_val_idx = None
    test_idx = None
    for train_val_idx, test_idx in shuffler.split(
            np.empty((labels.size, 1)), y=y, groups=groups):
        counter += 1
    assert counter == 1
    # make sure that they cover everything.
    assert np.array_equal(
        np.sort(np.concatenate((train_val_idx, test_idx))),
        np.arange(labels.size))
    return np.sort(train_val_idx), np.sort(test_idx)


def split_stuff(x_all, y, idx_sets):
    result = []

    for idx in idx_sets:
        result.append(x_all[idx])
        result.append(y[idx])

    return tuple(result)


def combine_two_separate_datasets(
        *,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
):
    # this is for handling "transfer learning" where two datasets with completely different neurons are trained together
    assert x1.ndim == x2.ndim == 4
    assert x1.shape[1:] == x2.shape[1:]
    assert y1.ndim == y2.ndim == 2

    n1, n2 = x1.shape[0], x2.shape[0]
    assert y1.shape[0] == n1
    assert y2.shape[0] == n2

    m1, m2 = y1.shape[1], y2.shape[1]
    assert y1.dtype == y2.dtype
    assert x1.dtype == x2.dtype

    assert n1 > 0
    assert n2 > 0
    assert m1 > 0
    assert m2 > 0

    # for x, just concatenate
    x = np.concatenate([x1, x2], axis=0)
    assert x.shape == (n1 + n2,) + x1.shape[1:]

    # for y, we need to fill some NaN to a bigger (n1+n2, m1+m2) array
    y = np.full((n1 + n2, m1 + m2), fill_value=np.nan, dtype=y1.dtype)

    y[:n1, :m1] = y1
    y[n1:, m1:] = y2

    return x, y
