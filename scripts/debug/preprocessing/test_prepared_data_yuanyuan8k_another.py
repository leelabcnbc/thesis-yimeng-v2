"""compare prepared 8k data in terms of BOTH images and neural data

using another API.
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from thesis_v2 import dir_dict, join
from thesis_v2.data.prepared.yuanyuan_8k import get_data

# I will traverse every item in newly generated data,
# and then map them to older data.

ref_root = join(dir_dict['private_data'],
                'yuanyuan_8k', 'final')

data_maps = {
    'a': (
        '042318_043018_051018',
        '042318',
        '043018',
        '051018',
    ),
    'b': (
        '050718',
        '051118',
    ),
    'c': (
        '050918',
    ),
}

for group, case_list in data_maps.items():
    for case in case_list:
        print(f'{group}/{case} begin')
        old_data = loadmat(join(ref_root,
                                f'8000{group}_{case}_128.mat'))
        new_data = get_data(group, 256, 128, case.split('_'),
                            seed='legacy', read_only=True)
        old_data = tuple(
            old_data[k] for k in (
                'data_train', 'labels_train',
                'data_valid', 'labels_valid',
                'data_test', 'labels_test',
            )
        )

        assert len(new_data) == len(old_data) == 6
        for y1, y2 in zip(new_data[1::2], old_data[1::2]):
            assert np.array_equal(y1, y2)

        for x1, x2 in zip(new_data[::2], old_data[::2]):
            # print(x1.min(), x1.max(), x1.mean())
            # print(x2.min(), x2.max(), x2.mean())
            assert x1.shape == x2.shape
            # get per image pearsor corr
            pearson_list = [pearsonr(c1.ravel(),
                                     c2.ravel())[0] for (c1,
                                                         c2) in zip(x1,
                                                                    x2)]
            pearson_list = np.asarray(pearson_list)
            assert pearson_list.mean() >= 0.99
            if pearson_list.min() < 0.98:
                print(pearson_list.min(), 'check this out')
            assert pearson_list.min() >= 0.95
            assert pearson_list.shape == (len(x1),)
        print(f'{group}/{case} end')
