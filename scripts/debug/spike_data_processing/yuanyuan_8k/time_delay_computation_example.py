"""just try replicating the following line of final_automated.m

get_time_delay_automated('042318', '8000a', [1, 2, 3, 4, 5, 6]);
"""
from os.path import join

from scipy.io import loadmat
import numpy as np

from thesis_v2 import dir_dict
from thesis_v2.spike_data_processing.yuanyuan_8k.time_delay import find_delay
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k
from thesis_v2.spike_data_processing.yuanyuan_8k.io_8k import load_spike_count_and_meta_data


def main():
    data = load_spike_count_and_meta_data('042318')

    result = find_delay(
        time_delays_to_try=config_8k.time_delays_to_try,
        spike_count_list=data['spike_count_list'],
        param_id_list=data['param_id_list'],
        mapping_record_paras=data['record_paras'],
        frame_per_image=config_8k.frame_per_image,
        duration_per_frame=config_8k.duration_per_frame,
        normaliztion_config='legacy',
        delay_metric='legacy',
        extraction_length=config_8k.extration_length_for_finding_time_delay,
    )

    best_delay = result['best_delay'] - 100
    best_correlation = result['best_correlation']

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
