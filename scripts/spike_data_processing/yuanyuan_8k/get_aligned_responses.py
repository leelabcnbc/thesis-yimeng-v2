"""replicating all get_aligned_responses_automated.m
"""
from os.path import join

import h5py
import numpy as np

from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k
from thesis_v2.spike_data_processing.yuanyuan_8k.io_8k import load_spike_count_and_meta_data
from thesis_v2.spike_data_processing.yuanyuan_8k.response_extraction import extract_response_given_time_delay


def main():
    save_file = join(config_8k.result_root_dir, 'responses.hdf5')

    delay_file = join(config_8k.result_root_dir, 'time_delay.hdf5')

    with h5py.File(save_file) as f:
        for prefix in config_8k.get_file_names(flat=False).keys():
            if prefix not in f:
                print(f'processing {prefix}')

                with h5py.File(delay_file, 'r') as f_delay:
                    time_delays = f_delay[prefix]['best_delay'][()]

                data = load_spike_count_and_meta_data(prefix)

                result = extract_response_given_time_delay(
                    time_delays=time_delays,
                    spike_count_list=data['spike_count_list'],
                    param_id_list=data['param_id_list'],
                    mapping_record_paras=data['record_paras'],
                    frame_per_image=config_8k.frame_per_image,
                    duration_per_frame=config_8k.duration_per_frame,
                    normaliztion_config='legacy',
                    extraction_length=config_8k.extration_length_for_response_computation,
                )

                response_all = result['response_all']
                response_mean = result['response_mean']

                assert response_all.ndim == 3
                assert response_mean.ndim == 2
                assert np.all(np.isfinite(response_all))
                assert np.all(np.isfinite(response_mean))

                g = f.create_group(prefix)
                g.create_dataset(name='response_mean', data=response_mean)
                g.create_dataset(name='response_all', data=response_all)
            else:
                print(f'done {prefix} before')


if __name__ == '__main__':
    main()
