"""
this should be run with thesis-v2 code
"""  # noqa: E501

from sys import argv

import numpy as np

from thesis_v2.data.prepared.crcns_pvc8 import natural_data

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v2.training_extra.maskcnn_like.training import (train_one,
                                                            partial)

from thesis_v2.models.maskcnn_polished.builder import (gen_maskcnn_polished, load_modules)

load_modules()

# keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
datasets_raw = natural_data('large', 144, 4, 0, scale=1 / 50, shuffle_type='legacy')

datasets_raw = {
    'X_train': datasets_raw[0].astype(np.float32),
    'y_train': datasets_raw[1],
    'X_val': datasets_raw[2].astype(np.float32),
    'y_val': datasets_raw[3],
    'X_test': datasets_raw[4].astype(np.float32),
    'y_test': datasets_raw[5],
}


def gen_cnn_partial(input_size, n):
    return gen_maskcnn_polished(input_size=input_size,
                                num_neuron=n,
                                out_channel=16,  # (try, 8, 16, 32, 48)
                                kernel_size_l1=7,  # (try 5,9,13)
                                kernel_size_l23=3,
                                act_fn='softplus',
                                pooling_ksize=3,  # (try, 1,3,5,7)
                                pooling_type='avg',  # try (avg, max)  # looks that max works well here?
                                num_layer=2,
                                )


def do_something(note):
    opt_config_partial = partial(get_maskcnn_v1_opt_config,
                                 scale=0.00005 * 221, smoothness=0.00002,
                                 group=0.0)

    results = []
    for seed in range(3):
        results.append(train_one(arch_json_partial=gen_cnn_partial,
                                 opt_config_partial=opt_config_partial,
                                 datasets=datasets_raw,
                                 key=f'debug/maskcnn_polished/v2_crcns_code/{note}/{seed}',
                                 show_every=1000,
                                 model_seed=seed,
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
# I get [0.7311413002904185, 0.7317113982875962, 0.7332677967273272]
# matching what's on GitHub.

# for image-v2
# (yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg)
# I get [0.7366746361978453, 0.7258181653545992, 0.7373553654726814]
