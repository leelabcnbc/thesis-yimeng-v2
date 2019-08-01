"""just try replicating the following line of final_automated.m

get_time_delay_automated('042318', '8000a', [1, 2, 3, 4, 5, 6]);
"""
from os.path import join

from scipy.io import loadmat
import h5py
import numpy as np

from thesis_v2 import dir_dict
from thesis_v2.spike_data_processing.yuanyuan_8k.response_extraction import extract_response
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k


def main():

    spike_count_dir = join(config_8k.result_root_dir, 'spike_count')
    n_neuron = 29

    spike_count_list = []
    with h5py.File(join(spike_count_dir, '042318.hdf5'), 'r') as f:
        # get all spike counts
        assert f.attrs['total'] == 6
        for idx in range(6):
            spike_count_list.append(f[str(idx)][()])

    param_id_list = [1, 2, 3, 4, 5, 6]

    assert len(spike_count_list) == len(param_id_list)

    # get data one by one.
    correlations_all = []

    mapping_file = loadmat(join(dir_dict['private_data'],
                                'yuanyuan_8k_preprocessing',
                                'Record_paras_Mar072018_RP.mat'
                                ))['Record_paras']

    for time_delay_idx, time_delay_this in enumerate(config_8k.time_delays_to_try):
        response_all_this_delay = []
        for spike_count, param_id in zip(spike_count_list, param_id_list):
            # get the shuffle idx
            assert spike_count.shape == (n_neuron, 500, 1100)
            sort_idx = mapping_file[param_id, 0][0, 1].ravel()
            assert sort_idx.shape == (8000,)
            assert np.array_equal(np.sort(sort_idx), np.arange(1, 8001))
            sort_idx = np.argsort(sort_idx)

            response_all_this_delay.append(
                extract_response(
                    spike_count=spike_count,
                    sort_index=sort_idx,
                    # convert to int, so that round() method does not get overloaded (producing a float)
                    time_delay=time_delay_this,
                    num_frame=config_8k.frame_per_image,
                    # althougth yuanyuan's code writes 60, it's effectively 61
                    extraction_length=48,
                    frame_time=config_8k.duration_per_frame,
                    normaliztion_config='legacy',
                )
            )
        response_all_this_delay = np.asarray(response_all_this_delay)
        # num_trial x num_neuron x num_image
        assert response_all_this_delay.shape == (6, n_neuron, 8000)

        # then compute correlation.
        correlations_this = []
        for idx_neuron in range(response_all_this_delay.shape[1]):
            corr_this = np.corrcoef(response_all_this_delay[:, idx_neuron], rowvar=True)
            # take upper triangular.
            # TODO: check number of NaNs.
            correlations_this.append(np.nanmean(corr_this[np.triu_indices(6, 1)]))
        correlations_all.append(correlations_this)

    correlations_all = np.asarray(correlations_all)
    print(correlations_all.shape)
    # best offset index
    best_delay_index = np.argmax(correlations_all, axis=0)
    best_correlation = correlations_all[best_delay_index, np.arange(n_neuron)]
    best_delay = np.asarray(config_8k.time_delays_to_try)[best_delay_index] - 100
    print(best_delay)
    print(best_correlation)

    # load reference file.
    ref_data = loadmat(join(dir_dict['private_data'], 'yuanyuan_8k', 'delay', 'delay_042318.mat'))
    best_delay_ref = ref_data['time_delay'].ravel()
    assert np.array_equal(best_delay_ref, best_delay)
    best_correlation_ref = ref_data['neurons'][2]
    assert best_correlation_ref.shape == best_correlation.shape
    print(abs(best_correlation_ref - best_correlation).max())
    assert abs(best_correlation_ref - best_correlation).max() < 1e-8


if __name__ == '__main__':
    main()
