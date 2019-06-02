from typing import Union

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from ..training.training_aux import infinite_n_batch_loader


# noinspection PyPep8Naming
def _check_dataset_shape(X: np.ndarray, y: Union[np.ndarray, tuple]):
    if not (X is None and y is None):
        if not isinstance(y, np.ndarray):
            assert isinstance(y, tuple)
            y_additional = y[1:]
            y = y[0]
        else:
            y_additional = ()
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert X.ndim == 4 and y.ndim == 2
        # print(X.shape, y.shape)
        assert X.shape[0] == y.shape[0] and X.shape[0] > 0

        for y_a in y_additional:
            assert isinstance(y_a, np.ndarray) and y_a.shape[0] == X.shape[0]

        # other than y[0], we keep original data types.
        assert X.dtype == np.float32
        # noinspection PyUnresolvedReferences
        x_ret = torch.from_numpy(X)
        assert y.dtype == np.float32
        # noinspection PyUnresolvedReferences
        y_ret = torch.from_numpy(y)
        # noinspection PyCallingNonCallable
        return (
            x_ret,
            (y_ret,) + tuple(torch.tensor(y_a) for y_a in y_additional)
        )
    else:
        return None, None


# noinspection PyPep8Naming
def generate_datasets(*,
                      X_train: np.ndarray, y_train: Union[np.ndarray, tuple],
                      X_val: np.ndarray = None,
                      y_val: Union[np.ndarray, tuple] = None,
                      X_test: np.ndarray = None,
                      y_test: Union[np.ndarray, tuple] = None,
                      batch_size=256, per_epoch_train=True,
                      shuffle_train=True,
                      ):
    X_train, y_train = _check_dataset_shape(X_train, y_train)
    X_test, y_test = _check_dataset_shape(X_test, y_test)
    X_val, y_val = _check_dataset_shape(X_val, y_val)

    if per_epoch_train:
        assert X_train.size()[0] >= batch_size

    dataset_train = TensorDataset(X_train, *y_train)
    if per_epoch_train:
        # since we drop_last, that's why X_train has to be long enough.
        dataset_train = DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=shuffle_train,
                                   drop_last=True)
        dataset_train = infinite_n_batch_loader(dataset_train, n=1)
    else:
        dataset_train = DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=False)

    if X_test is not None and y_test is not None:
        dataset_test = DataLoader(TensorDataset(X_test, *y_test),
                                  batch_size=batch_size)
    else:
        dataset_test = None
    if X_val is not None and y_val is not None:
        dataset_val = DataLoader(TensorDataset(X_val, *y_val),
                                 batch_size=batch_size)
    else:
        dataset_val = None

    return {
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test,
    }


def get_resp_train(datasets):
    if not isinstance(datasets['y_train'], np.ndarray):
        assert isinstance(datasets['y_train'], tuple)
        resp_train = datasets['y_train'][0]
    else:
        resp_train = datasets['y_train']

    return resp_train
