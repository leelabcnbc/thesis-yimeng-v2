"""a modification of
results in the last cells of https://github.com/leelabcnbc/thesis-yimeng-v1/blob/master/results_ipynb/crcns_pvc-8_large/maskcnn_polished_basic.ipynb

In particular, I tried to reproduce the out_channel=16, pooling_ksize=3 version.

this should be run with thesis-v1 code (as well as the preprocessed data using v1 code); ./setup_env_variables.sh
for v1 should be used.
"""  # noqa: E501

from sys import argv

from thesis_v1.models.utils.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v1.models.utils.maskcnn_like.training import (train_one,
                                                          partial)

from thesis_v1.data.prepared.crcns_pvc8 import natural_data_legacy

from thesis_v1.models.maskcnn_polished.builder import (gen_maskcnn_polished, load_modules)

load_modules()


def print_dataset_info(datasets_this):
    for x in datasets_this:
        print(x.dtype, x.shape, x.min(), x.max(), x.mean())


# keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
datasets_raw = natural_data_legacy('large', 144, 4, 0, scale=1 / 50)


def gen_cnn_partial(input_size, n):
    return gen_maskcnn_polished(input_size, n,
                                out_channel=16,
                                kernel_size_l1=7,
                                kernel_size_l23=3,
                                act_fn='softplus',
                                pooling_ksize=3,
                                pooling_type='avg',
                                num_layer=2)


def do_something(note):
    opt_config_partial = partial(get_maskcnn_v1_opt_config,
                                 scale=0.00005 * 221, smoothness=0.00002,
                                 group=0.0, legacy=False)

    results = []
    for seed in range(3):
        results.append(train_one(gen_cnn_partial, opt_config_partial,
                                 datasets_raw,
                                 key=f'debug/maskcnn_polished/v1_crcns_code/{note}/{seed}',
                                 show_every=1000, legacy_corr=True,
                                 model_seed=seed, return_model=False))

    print([r['stats_best']['stats']['test']['corr_mean'] for r in results])


if __name__ == '__main__':
    # randomly pick a small layer.
    assert len(argv) == 2
    # argv[1] can be used to denote which image
    # say, `image-v1` or `image-v2`
    do_something(argv[1])

# the reason that here the code reports 29011 parameters but ipynb gives 29077 is whether some BN stuffs count
# as parameters; the difference is (29077-29011)=66=1*2 + 16*2 + 16*2 (three BN layers' var/mean).

# on 2080ti machine,
# for image-v1,
# (leelabcnbc_yimeng-thesis_cu100-2019-02-07-8c53c104a86e.simg)
# I get [0.7311413002904185, 0.7317113982875962, 0.7332677967273272]

# for image-v2
# (yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg)
# I get [0.7366746361978453, 0.7258181653545992, 0.7373553654726814]

# quite close to what's on GitHub, which gives 0.732854 over 5 seeds.
