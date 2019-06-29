"""
this should be run with thesis-v2 code
"""  # noqa: E501

from sys import argv

import numpy as np

from thesis_v2.data.prepared.yuanyuan_8k import get_data

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v2.training_extra.maskcnn_like.training import (train_one,
                                                            partial)

from thesis_v2.models.maskcnn_polished_with_local_pcn.builder import (gen_maskcnn_polished_with_local_pcn,
                                                                      load_modules)

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


def gen_cnn_partial(input_size, n):
    return gen_maskcnn_polished_with_local_pcn(input_size=input_size,
                                               num_neuron=n,
                                               out_channel=16,  # (try, 8, 16, 32, 48)
                                               kernel_size_l1=9,  # (try 5,9,13)
                                               kernel_size_l23=3,
                                               act_fn='softplus',
                                               pooling_ksize=3,  # (try, 1,3,5,7)
                                               pooling_type='avg',  # try (avg, max)  # looks that max works well here?
                                               num_layer=2,
                                               pcn_bn=True,
                                               pcn_bn_post=True,
                                               pcn_bypass=True,
                                               pcn_cls=1,
                                               pcn_final_act=False,
                                               pcn_no_act=False,
                                               pcn_b0_init=1.0,
                                               pcn_tied=False,
                                               bn_locations_legacy=True,
                                               )


def do_something(note):
    opt_config_partial = partial(get_maskcnn_v1_opt_config,
                                 scale=0.01,
                                 smoothness=0.00005,
                                 group=0.0)

    results = []
    for seed in range(3):
        results.append(train_one(arch_json_partial=gen_cnn_partial,
                                 opt_config_partial=opt_config_partial,
                                 datasets=datasets_raw,
                                 key=f'debug/maskcnn_polished_with_local_pcn/v2_code/{note}/{seed}',
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
# I get [0.5168265104293823, 0.522405743598938, 0.5129156112670898]
# matching what's on GitHub.

# for image-v2
# (yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg)
# I get [0.516857922077179, 0.5226259231567383, 0.5128955841064453]
