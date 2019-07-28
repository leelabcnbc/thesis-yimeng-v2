"""same effect as get_spike_counts.m in yuanyuan's repo"""

from os.path import join, exists
from os import makedirs

from scipy.io import loadmat
import h5py
import numpy as np

from thesis_v2.spike_data_processing.yuanyuan_8k.config_8k import get_file_names, good_channel_unit, result_root_dir
from thesis_v2.spike_data_processing.yuanyuan_8k.spike_count import cdttable_to_spike_count
from thesis_v2 import dir_dict


def main():
    save_dir = join(result_root_dir, 'spike_count')
    if not exists(save_dir):
        makedirs(save_dir)

    for prefix, fnames in get_file_names(flat=False).items():
        # create hdf5 file
        with h5py.File(join(save_dir, f'{prefix}.hdf5')) as f:
            for fname_idx, fname in enumerate(fnames):
                if str(fname_idx) not in f:
                    print(f'working on {prefix}/{fname}')
                    # load cdttable
                    cdttable = loadmat(join(
                        dir_dict['private_data'], 'yuanyuan_8k_raw', 'cdt',
                        fname,
                    ))['CDTTables'][0, 0][0, 0]
                    spike_count = cdttable_to_spike_count(
                        cdttable=cdttable,
                        neurons_to_extract=good_channel_unit(filename=prefix, is_prefix=True),
                        start_time=400,
                        end_time=1500,
                        num_condition_debug=500,
                    )
                    f.create_dataset(name=str(fname_idx), data=spike_count, compression='gzip')
                    f[str(fname_idx)].attrs['name'] = np.string_(fname)
                    f.flush()
                else:
                    print('{prefix}/{fname} done before')
                f.attrs['total'] = np.int64(len(fnames))


if __name__ == '__main__':
    main()
