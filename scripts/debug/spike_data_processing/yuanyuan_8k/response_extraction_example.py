"""just try replicating the following line of final_automated.m

get_aligned_responses_automated('042318', '8000a', [1, 2, 3, 4, 5, 6]);
"""
from os.path import join

from scipy.io import loadmat
from scipy.stats import pearsonr
import h5py
import numpy as np

from thesis_v2 import dir_dict
from thesis_v2.spike_data_processing.yuanyuan_8k.response_extraction import extract_response
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k


def main():
    time_delay_dir = join(dir_dict['private_data'], 'yuanyuan_8k', 'delay')
    spike_count_dir = join(config_8k.result_root_dir, 'spike_count')
    time_delays = loadmat(join(time_delay_dir, 'delay_042318.mat'))['time_delay'].ravel()
    assert time_delays.shape == (29,)

    spike_count_list = []
    with h5py.File(join(spike_count_dir, '042318.hdf5'), 'r') as f:
        # get all spike counts
        assert f.attrs['total'] == 6
        for idx in range(6):
            spike_count_list.append(f[str(idx)][()])

    param_id_list = [1, 2, 3, 4, 5, 6]

    assert len(spike_count_list) == len(param_id_list)

    # get data one by one.
    response_all = []

    mapping_file = loadmat(join(dir_dict['private_data'],
                                'yuanyuan_8k_preprocessing',
                                'Record_paras_Mar072018_RP.mat'
                                ))['Record_paras']

    for spike_count, param_id in zip(spike_count_list, param_id_list):
        # get the shuffle idx
        sort_idx = mapping_file[param_id, 0][0, 1].ravel()
        assert sort_idx.shape == (8000,)
        assert np.array_equal(np.sort(sort_idx), np.arange(1, 8001))
        sort_idx = np.argsort(sort_idx)

        response_this = []

        for neuron_idx, time_delay_this in enumerate(time_delays):
            response_this.append(
                extract_response(
                    spike_count=spike_count,
                    sort_index=sort_idx,
                    # convert to int, so that round() method does not get overloaded (producing a float)
                    time_delay=int(time_delay_this + 100),
                    num_frame=config_8k.frame_per_image,
                    # althougth yuanyuan's code writes 60, it's effectively 61
                    extraction_length=61,
                    frame_time=config_8k.duration_per_frame,
                    neuron_list=[neuron_idx, ],
                    normaliztion_config='legacy',
                )
            )
        response_this = np.concatenate(response_this, axis=0)
        response_all.append(response_this)
    response_all = np.asarray(response_all).mean(axis=0)
    # print(response_all.shape)

    response_all_debug = loadmat(
        join(dir_dict['private_data'], 'yuanyuan_8k', 'resp', 'resp_042318.mat')
    )['resp'].T

    # print(response_all.shape, response_all_debug.shape)
    assert response_all.shape == response_all_debug.shape
    assert np.array_equal(response_all, response_all_debug)


if __name__ == '__main__':
    main()
