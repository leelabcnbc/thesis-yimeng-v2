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
