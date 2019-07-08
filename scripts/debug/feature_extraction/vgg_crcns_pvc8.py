# check that CNN feature extraction works.
# I only check vgg 8k

# when I ran this file, only
#   'vgg11',
#   'vgg11_bn',
# networks were tested, and only scale=quarter was tested.

from functools import partial

import h5py
import numpy as np

from thesis_v2 import dir_dict, join

file_to_save_input = join(dir_dict['datasets'],
                          'cnn_feature_extraction_input',
                          'crcns_pvc8.hdf5')

file_to_save_feature = join(dir_dict['features'],
                            'cnn_feature_extraction',
                            'crcns_pvc8',
                            'vgg.hdf5'
                            )

file_to_save_input_ref = join(dir_dict['root'],
                              '..', 'thesis-yimeng-v1',
                              'results', 'datasets',
                              'cnn_feature_extraction_input_vgg.hdf5')
file_to_save_feature_ref = join(dir_dict['root'],
                                '..', 'thesis-yimeng-v1',
                                'results', 'features',
                                'cnn_feature_extraction_vgg.hdf5')


def callback(name, obj, state_dict):
    if isinstance(obj, h5py.Dataset):
        assert obj.ndim >= 1 and obj.shape[0] >= 20
        data_old = obj[()]
        # print(name)
        prefix = 'crcns_pvc-8_'
        assert name.startswith(prefix)

        name_split = name.split('/')
        assert len(name_split) == state_dict['path_length']

        if name_split[1] not in {'vgg11', 'vgg11_bn', 'vgg16', 'vgg16_bn'}:
            return

        # get data old.
        f_n = state_dict['f_new']

        if state_dict.get('remove_network', False):
            name_to_fetch = name_split[0][len(prefix):] + '/' + name_split[2]
        else:
            name_to_fetch = name[len(prefix):]

        data_new = f_n[name_to_fetch][()]
        print(name, data_new.shape, np.array_equal(data_new, data_old))

        assert data_new.shape == data_old.shape
        assert np.allclose(data_new, data_old, atol=1e-5)


with h5py.File(file_to_save_input_ref, 'r') as f_ref:
    with h5py.File(file_to_save_input, 'r') as f_new:
        f_ref.visititems(partial(callback, state_dict={
            'f_new': f_new,
            'remove_network': True,
            'path_length': 3,
        }))

with h5py.File(file_to_save_feature, 'r') as f_new:
    with h5py.File(file_to_save_feature_ref, 'r') as f_ref:
        f_ref.visititems(partial(callback, state_dict={
            'f_new': f_new,
            'path_length': 4,
        }))
