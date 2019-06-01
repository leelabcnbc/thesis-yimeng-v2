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
                          'yuanyuan_8k_a.hdf5')

file_to_save_feature = join(dir_dict['features'],
                            'cnn_feature_extraction',
                            'yuanyuan_8k_a',
                            'vgg.hdf5'
                            )

file_to_save_input_ref = join(dir_dict['root'],
                              '..', 'thesis-yimeng-v1',
                              'results', 'datasets',
                              'cnn_feature_extraction_input_8k_fullsize.hdf5')
file_to_save_feature_ref = join(dir_dict['root'],
                                '..', 'thesis-yimeng-v1',
                                'results', 'features',
                                'cnn_feature_extraction_8k_vgg_fullsize.hdf5')


def callback(name, obj, state_dict):
    if isinstance(obj, h5py.Dataset):
        # to speed up, only pick 1/20.
        # otherwise, too slow.
        assert obj.ndim >= 1 and obj.shape[0] >= 20
        data_new = obj[::20]

        # get data old.
        f_old = state_dict['f_old']
        data_old = f_old[name][::20]
        print(name, data_new.shape, np.array_equal(data_new, data_old))

        assert data_new.shape == data_old.shape
        assert np.allclose(data_new, data_old, atol=1e-5)


with h5py.File(file_to_save_input_ref, 'r') as f_ref:
    with h5py.File(file_to_save_input, 'r') as f_new:
        f_new.visititems(partial(callback, state_dict={
            'f_old': f_ref,
        }))

with h5py.File(file_to_save_feature_ref, 'r') as f_ref:
    with h5py.File(file_to_save_feature, 'r') as f_new:
        f_new.visititems(partial(callback, state_dict={
            'f_old': f_ref,
        }))
