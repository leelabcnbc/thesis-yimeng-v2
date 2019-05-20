from functools import partial
import h5py
import numpy as np
from numpy.testing import assert_array_equal

from thesis_v2 import dir_dict, join


def allclose_comparator(rtol=1e-5, atol=1e-8):
    return lambda x, y: x.shape == y.shape and np.allclose(x, y,
                                                           rtol=rtol,
                                                           atol=atol)


def load_hdf5_data_one_piece(file_name, obj_name, attr_name):
    with h5py.File(file_name, 'r') as f_old:
        obj_old = f_old[obj_name]
        if attr_name is None:
            data_old = obj_old[...]
        else:
            data_old = np.asarray(obj_old.attrs[attr_name])
    return data_old


def callback(name, obj, mapping):
    if isinstance(obj, h5py.Dataset):
        if name not in mapping:
            raise RuntimeError(f'NOT IMPLEMENTED: {name}')
        else:
            print(name)
            # get that data.
            mapped = mapping[name]
            mapped += (None,) * (4 - len(mapped))
            fname_old, obj_name, attr_name, comparator = mapped

            data_old = load_hdf5_data_one_piece(
                join(dir_dict['debug_data'],
                     fname_old),
                obj_name, attr_name
            )

            data_new = obj[...]
            if comparator is None:
                # this compares NaN as well.
                assert_array_equal(data_new, data_old)
            else:
                assert comparator(data_new, data_old)


def check_one_file(fname, mapping):
    print(f'working on {fname} begin')
    with h5py.File(fname, 'r') as f_new:
        f_new.visititems(partial(callback, mapping=mapping))
    print(f'working on {fname} end')
