"""make sure that newly prepared CRCNS PVC-8 raw data
match those prepared in the before.
"""
import numpy as np
from scipy.io import loadmat
from thesis_v2 import dir_dict, join
from thesis_v2.data.raw import load_data
# noinspection PyProtectedMember
from thesis_v2.data.raw.yuanyuan_8k import _neural_data_dates


# I will traverse every item in newly generated data,
# and then map them to older data.


def test_images():
    yuanyuan_ref_img_root = join(dir_dict['private_data'], 'yuanyuan_8k',
                                 'img')
    data_maps = {
        'yuanyuan_8k_images': {
            'a': 'img_8000a.mat',
            'b': 'img_8000b.mat',
            'c': 'img_8000c.mat',
        },

    }

    for key, mapping in data_maps.items():
        print(f'{key} begin')
        for dataname, oldfile in mapping.items():
            print(f'{key}/{dataname} begin')
            data_new = load_data(key, dataname)['images']
            data_old = loadmat(join(yuanyuan_ref_img_root,
                                    oldfile))['images'].ravel()
            assert len(data_new) == len(data_old) == 8000
            for img_new, img_old in zip(data_new, data_old):
                assert np.array_equal(img_new, img_old)
                assert img_new.shape == (
                    400, 400) and img_new.dtype == np.uint8
            print(f'{key}/{dataname} end')
        print(f'{key} end')


def test_neural():
    yuanyuan_ref_resp_root = join(dir_dict['private_data'], 'yuanyuan_8k',
                                  'resp')
    for key in _neural_data_dates:
        print(f'{key} begin')
        data_new = load_data('yuanyuan_8k_neural', key)['resp']
        data_old = loadmat(join(yuanyuan_ref_resp_root,
                                f'resp_{key}.mat'))['resp']
        assert np.array_equal(data_new, data_old)
        assert data_new.shape[0] == 8000 and data_new.ndim == 2
        print(f'{key} end')


test_images()
test_neural()
