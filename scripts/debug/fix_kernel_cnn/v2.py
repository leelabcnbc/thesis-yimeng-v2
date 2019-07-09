"""
this should be run with thesis-v2 code
"""  # noqa: E501

from sys import argv

import numpy as np

from thesis_v2.data.prepared.yuanyuan_8k import get_data

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v2.training_extra.maskcnn_like.training import (train_one,
                                                            partial)

from thesis_v2.models.fix_kernel_cnn.builder import (gen_fkcnn, load_modules)

load_modules()

# keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
datasets_raw = get_data('a', 200, 50, ('042318', '043018', '051018'), scale=0.5)

datasets_raw = {
    'X_train': datasets_raw[0].astype(np.float32),
    'y_train': datasets_raw[1],
    'X_val': datasets_raw[2].astype(np.float32),
    'y_val': datasets_raw[3],
    'X_test': datasets_raw[4].astype(np.float32),
    'y_test': datasets_raw[5],
}

weight = np.load('filters.npy')
#manually select 24 out of total 79 kernels
weight = weight[[0,3,5,8,9,11,14,17,18,20,23,25,27,32,36,37,40,44,53,57,58,64,65,74],:,:]
weight = weight.reshape(24,1,9,9)

def gen_cnn_partial(input_size, n):
    return gen_fkcnn(input_size=input_size,
                         num_neuron=n,
                         kernel_size_l1=9, out_channel_l1=24, kernel_size_l23=5, out_channel_l23=24,
                         factored_constraint=None,
                         act_fn='relu',
                         do_init=True,
                         pooling_type='max',
                         pooling_ksize=2,
                         num_layer=3,
                         bn_before_act=True,
                         bn_after_fc=False
                         )


def do_something(note):
    opt_config_partial = partial(get_maskcnn_v1_opt_config,
                                 scale=0.1,
                                 smoothness=0.000005,
                                 group=0.0,
                                 first_layer_no_learning=True)

    results = []
    for seed in range(3):
        results.append(train_one(arch_json_partial=gen_cnn_partial,
                                 opt_config_partial=opt_config_partial,
                                 datasets=datasets_raw,
                                 key=f'debug/FKCNN/v2_code/{note}/{seed}',
                                 show_every=1000,
                                 model_seed=seed,
                                 use_fkcnn=True,
                                 return_model=False)
                       )

    print([r['stats_best']['stats']['test']['corr_mean'] for r in results])


if __name__ == '__main__':
    # randomly pick a small layer.
    assert len(argv) == 2
    # argv[1] can be used to denote which image
    # say, `image-v1` or `image-v2`
    do_something(argv[1])

# on 2080ti machine,
# for image-v1,
# (leelabcnbc_yimeng-thesis_cu100-2019-02-07-8c53c104a86e.simg)
# I get [0.513634443283081, 0.5110703706741333, 0.512565016746521]
# matching what's on GitHub.

# for image-v2
# (yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg)
# I get [0.5136207938194275, 0.5107940435409546, 0.5097903609275818]
