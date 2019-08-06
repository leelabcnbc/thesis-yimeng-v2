"""just try replicating the following line of final_automated.m

get_time_delay_automated('042318', '8000a', [1, 2, 3, 4, 5, 6]);
"""
from os.path import join

from scipy.io import loadmat
import numpy as np
import h5py

from thesis_v2 import dir_dict
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k


def main():
    for prefix in config_8k.get_file_names(flat=False).keys():
        print(f'check {prefix}')
        with h5py.File(join(config_8k.result_root_dir, 'time_delay.hdf5'), 'r') as f:
            best_delay = f[prefix]['best_delay'][()] - 100
            best_correlation = f[prefix]['best_correlation'][()]

        # load reference file.
        ref_data = loadmat(join(dir_dict['private_data'], 'yuanyuan_8k', 'delay', f'delay_{prefix}.mat'))
        best_delay_ref = ref_data['time_delay'].ravel()
        assert np.array_equal(best_delay_ref, best_delay)
        best_correlation_ref = ref_data['neurons'][2]
        assert best_correlation_ref.shape == best_correlation.shape
        print(abs(best_correlation_ref - best_correlation).max())
        assert abs(best_correlation_ref - best_correlation).max() < 1e-8


if __name__ == '__main__':
    main()
