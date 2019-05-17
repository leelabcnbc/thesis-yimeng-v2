"""this module processes raw data and stores them centrally
under a folder in HDF5.
"""

from os import makedirs
from ... import dir_dict, abspath, join
from .. import make_register_data_wrapper, save_data_helper, load_data_helper

dir_root = abspath(join(dir_dict['datasets'], 'raw'))
makedirs(dir_root, exist_ok=True)

_register_data_store = {}

register_data = make_register_data_wrapper(_register_data_store)


def save_data():
    # call this stuff to process all raw data.
    from . import crcns_pvc8
    from . import yuanyuan_8k
    save_data_helper(_register_data_store, dir_root)


def load_data(key, dataname):
    # call this stuff to process all raw data.
    return load_data_helper(key, dataname, dir_root)
