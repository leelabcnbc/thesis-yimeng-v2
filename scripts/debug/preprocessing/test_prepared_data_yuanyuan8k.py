"""compare prepared 8k data in terms of images

due to difference in image processing, it's expected to have
some differences.
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from thesis_v2 import dir_dict, join
from thesis_v2.data.prepared import dir_root
from thesis_v2.data import load_data_lazy_helper

# I will traverse every item in newly generated data,
# and then map them to older data.

yuanyuan_ref_img_root = join(dir_dict['private_data'],
                             'yuanyuan_8k', 'final')

data_maps = {
    'yuanyuan_8k_images': {
        # any 8000a should be fine
        'groupa/keep256/size128': (
            '8000a_042318_043018_051018_128.mat',
            '8000a_042318_128.mat',
            '8000a_043018_128.mat',
            '8000a_051018_128.mat',
        ),
        # any 8000b should be fine
        'groupb/keep256/size128': (
            '8000b_050718_128.mat',
            '8000b_051118_128.mat',
        ),
        # any 8000c should be fine
        'groupc/keep256/size128': (
            '8000c_050918_128.mat',
        ),
    },
}

idx_set = loadmat(join(dir_dict['private_data_supp'], 'yuanyuan_8k_idx.mat'))

for key, mapping in data_maps.items():
    print(f'{key} begin')
    for dataname, oldfile_list in mapping.items():
        print(f'{key}/{dataname} begin')
        data_new = load_data_lazy_helper(dataname, None,
                                         fname=join(dir_root, key + '.hdf5'))
        data_new = {
            'data_train': data_new[
                idx_set['I_train'].ravel().astype(np.bool_)],
            'data_valid': data_new[
                idx_set['I_valid'].ravel().astype(np.bool_)],
            'data_test': data_new[idx_set['I_test'].ravel().astype(np.bool_)],
        }
        for oldfile in oldfile_list:
            print(oldfile)
            data_old = loadmat(join(yuanyuan_ref_img_root, oldfile),
                               variable_names=('data_train',
                                               'data_valid',
                                               'data_test'))
            for fieldname in data_new.keys():
                xnew = data_new[fieldname]
                xold = data_old[fieldname]
                assert xnew.shape == xold.shape
                # get per image pearsor corr
                pearson_list = [pearsonr(x1.ravel(),
                                         x2.ravel())[0] for (x1,
                                                             x2) in zip(xnew,
                                                                        xold)]
                pearson_list = np.asarray(pearson_list)
                assert pearson_list.mean() >= 0.99
                if pearson_list.min() < 0.98:
                    print(pearson_list.min(), 'check this out')
                assert pearson_list.min() >= 0.95
                assert pearson_list.shape == (len(xnew),)

        print(f'{key}/{dataname} end')
    print(f'{key} end')
