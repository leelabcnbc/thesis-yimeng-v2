from os.path import join
from functools import partial
from typing import Dict, Callable, Union
import numpy as np
import h5py

_type_spec = Dict[str, Callable]
_type_store = Dict[str, _type_spec]


def _register_data(key: str,
                   spec: _type_spec,
                   *, store: _type_store):
    assert key not in store
    assert type(key) is str and key.lower() == key and ('/' not in key)
    assert type(spec) is dict
    for k, v in spec.items():
        assert type(k) is str and k.lower() == k and ('/' not in k)
        assert callable(v)

    # mapping from name to callable to generate dataset.
    store[key] = spec


def make_register_data_wrapper(store: _type_store):
    return partial(_register_data, store=store)


def save_data_helper(store: _type_store,
                     root_dir: str):
    for key, spec in store.items():
        # go through each file.
        print(f'processing {key} begin')
        filename = join(root_dir, key + '.hdf5')
        with h5py.File(filename) as f:
            for dataname, datafn in spec.items():
                if dataname in f:
                    print(f'processed {key}/{dataname} before')
                else:
                    print(f'processing {key}/{dataname} begin')
                    data = datafn()

                    _save_data_one(f, dataname, data)

                    print(f'processing {key}/{dataname} end')

        print(f'processing {key} end')


def load_data_helper(key: str,
                     dataname: str, root_dir: str):
    filename = join(root_dir, key + '.hdf5')
    with h5py.File(filename, 'r') as f:
        return _load_data_one(f[dataname])


# https://stackoverflow.com/questions/53845024/defining-a-recursive-type-hint-in-python  # noqa: E501

# https://www.python.org/dev/peps/pep-0484/#forward-references

_type_data = Union[np.ndarray, Dict[str, '_type_data']]


def _save_data_one(grp: h5py.Group,
                   dataname: str, data: _type_data):
    # dataname can contain slashes, actually.
    # but you won't be able to retrieve it back in the same format
    # using load_data_one.
    # (unless this slashed-name only happens at the top level).
    if type(data) is np.ndarray:
        # rather than isinstance(data, np.ndarray)
        # faster and safer

        # for scalar, please first wrap that as np.asarray yourself.
        # compression saves a lot of space. so do it.
        grp.create_dataset(dataname, data=data,
                           compression="gzip")
        grp.file.flush()
        print(f'done {grp[dataname].name}')
    elif type(data) is dict:
        # rather than isinstance(data, dict)
        # faster and safer

        grp_next = grp.create_group(dataname)
        for name_next, data_next in data.items():
            _save_data_one(grp_next, name_next, data_next)
    else:
        raise TypeError


def _load_data_one(data: Union[h5py.Group, h5py.Dataset]) -> _type_data:
    if type(data) is h5py.Dataset:
        return data[()]  # this way, scalar data sets are returned as scalar.
        # not array(xxx).
        # this holds regardless whether you store data as .create_dataset(...,
        #     data=scalar)
        # or .create_dataset(..., data=np.array(scalar))
        # the `scalar` will be returned by `()` in both cases.
    elif isinstance(data, h5py.Group):
        # this can be h5py.Group or h5py.File
        data_ret = dict()
        for k, v in data.items():
            data_ret[k] = _load_data_one(v)
        return data_ret
