from os.path import join

import h5py
from scipy.io import loadmat

from .config_8k import (
    result_root_dir, imageset_mapping_dict,
    record_paras_file_mapping_dict,
    para_file_mapping_dict
)

from ... import dir_dict


def load_spike_count_and_meta_data(prefix, *, load_spike=True):
    spike_count_dir = join(result_root_dir, 'spike_count')

    spike_count_list = None
    if load_spike:
        spike_count_list = []
        with h5py.File(join(spike_count_dir, f'{prefix}.hdf5'), 'r') as f:
            # get all spike counts
            num_sessions = f.attrs['total']
            for idx in range(num_sessions):
                spike_count_list.append(f[str(idx)][()])

    param_id_list = para_file_mapping_dict[prefix]

    record_paras = loadmat(join(dir_dict['private_data'],
                                'yuanyuan_8k_preprocessing',
                                record_paras_file_mapping_dict[imageset_mapping_dict[prefix]],
                                ))['Record_paras']

    return {
        'spike_count_list': spike_count_list,
        'param_id_list': param_id_list,
        'record_paras': record_paras,
    }
