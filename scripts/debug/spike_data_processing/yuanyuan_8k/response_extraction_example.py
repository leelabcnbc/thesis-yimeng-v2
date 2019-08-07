"""just try replicating the following line of final_automated.m

get_aligned_responses_automated('042318', '8000a', [1, 2, 3, 4, 5, 6]);
"""
from os.path import join

from scipy.io import loadmat
import h5py
import numpy as np

from thesis_v2 import dir_dict
from thesis_v2.spike_data_processing.yuanyuan_8k.response_extraction import extract_response_given_time_delay
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k
from thesis_v2.spike_data_processing.yuanyuan_8k.io_8k import load_spike_count_and_meta_data


def main():
    delay_file = join(config_8k.result_root_dir, 'time_delay.hdf5')

    with h5py.File(delay_file, 'r') as f_delay:
        time_delays = f_delay['042318']['best_delay'][()]

    assert time_delays.shape == (29,)

    data = load_spike_count_and_meta_data('042318')

    record_paras = data['record_paras']
    spike_count_list = data['spike_count_list']
    param_id_list = data['param_id_list']

    # get data one by one.
    response_all = extract_response_given_time_delay(
        time_delays=time_delays,
        spike_count_list=spike_count_list,
        param_id_list=param_id_list,
        mapping_record_paras=record_paras,
        frame_per_image=config_8k.frame_per_image,
        duration_per_frame=config_8k.duration_per_frame,
        normaliztion_config='legacy',
        extraction_length=config_8k.extration_length_for_response_computation,
    )['response_mean']

    # print(response_all.shape)

    response_all_debug = loadmat(
        join(dir_dict['private_data'], 'yuanyuan_8k', 'resp', 'resp_042318.mat')
    )['resp'].T

    # print(response_all.shape, response_all_debug.shape)
    assert response_all.shape == response_all_debug.shape
    assert np.array_equal(response_all, response_all_debug)


if __name__ == '__main__':
    main()
