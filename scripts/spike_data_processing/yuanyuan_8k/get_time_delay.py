"""replicating all get_time_delay_automated.m
"""
from os.path import join

import h5py

from thesis_v2.spike_data_processing.yuanyuan_8k.time_delay import find_delay
from thesis_v2.spike_data_processing.yuanyuan_8k import config_8k
from thesis_v2.spike_data_processing.yuanyuan_8k.io_8k import load_spike_count_and_meta_data


def main():
    save_file = join(config_8k.result_root_dir, 'time_delay.hdf5')

    with h5py.File(save_file) as f:
        for prefix in config_8k.get_file_names(flat=False).keys():
            if prefix not in f:
                print(f'processing {prefix}')

                data = load_spike_count_and_meta_data(prefix)

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

                best_delay = result['best_delay']
                best_correlation = result['best_correlation']

                assert best_correlation.shape == best_delay.shape
                assert best_delay.ndim == 1

                g = f.create_group(prefix)
                g.create_dataset(name='best_delay', data=best_delay)
                g.create_dataset(name='best_correlation', data=best_correlation)
            else:
                print(f'done {prefix} before')


if __name__ == '__main__':
    main()
