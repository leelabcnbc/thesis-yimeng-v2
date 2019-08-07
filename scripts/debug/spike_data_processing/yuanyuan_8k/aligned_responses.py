from os.path import join

from scipy.io import loadmat
import h5py
import numpy as np


from thesis_v2 import dir_dict
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k


def main():
    for prefix in config_8k.get_file_names(flat=False).keys():
        print(f'check {prefix}')
        with h5py.File(join(config_8k.result_root_dir, 'responses.hdf5'), 'r') as f:
            response_mean = f[prefix]['response_mean'][()]

        # load reference file.
        ref_data = loadmat(join(dir_dict['private_data'], 'yuanyuan_8k', 'resp', f'resp_{prefix}.mat'))['resp'].T
        assert response_mean.shape == ref_data.shape
        print(abs(response_mean - ref_data).max())
        assert abs(response_mean - ref_data).max() < 1e-8
        # actually, they are all the same...
        assert np.array_equal(response_mean, ref_data)


if __name__ == '__main__':
    main()
