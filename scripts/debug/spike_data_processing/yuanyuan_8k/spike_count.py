# check `get_spioke_counts.m` has been implemented correctly

from os.path import join

import h5py
import numpy as np

from thesis_v2.spike_data_processing.yuanyuan_8k.config_8k import get_file_names, result_root_dir
from thesis_v2 import dir_dict


def main():
    save_dir = join(result_root_dir, 'spike_count')
    ref_dir = join(dir_dict['private_data'], 'yuanyuan_8k', 'spc')
    for prefix, fnames in get_file_names(flat=False).items():
        print(prefix)
        with h5py.File(
                join(save_dir, f'{prefix}.hdf5'), 'r'
        ) as f, h5py.File(
            join(ref_dir, f'spc_{prefix}.mat'), 'r'
        ) as f_ref:
            total_count = f.attrs['total']
            assert f_ref['spike_counts'].shape == (total_count, 1)
            assert total_count == len(fnames)
            for idx in range(total_count):
                print(f'checking {idx}/{total_count}')
                data = f[str(idx)][()]
                data_ref = f_ref[f_ref['spike_counts'][idx, 0]][()].T
                assert data.shape == data_ref.shape
                # print(data[0, 0, :200])
                # print(data_ref[0, 0, :200])
                assert np.array_equal(data, data_ref)
                assert f[str(idx)].attrs['name'].decode() == fnames[idx]


if __name__ == '__main__':
    main()
