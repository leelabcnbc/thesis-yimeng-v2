"""a modification of
https://github.com/leelabcnbc/thesis-yimeng-v1/blob/master/results_ipynb/yuanyuan_8k_a_3day/maskcnn_polished_lite_baseline.ipynb

this should be run with thesis-v1 code (as well as the preprocessed data using v1 code); ./setup_env_variables.sh
for v1 should be used.
"""  # noqa: E501

from sys import argv

from thesis_v1.data.prepared.yuanyuan_8k import get_data

from thesis_v1.models.utils.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v1.models.utils.maskcnn_like.training import (train_one,
                                                          partial)

from thesis_v1.models.maskcnn_polished.builder import (gen_maskcnn_polished, load_modules)

load_modules()


def print_dataset_info(datasets_this):
    for x in datasets_this:
        print(x.dtype, x.shape, x.min(), x.max(), x.mean())


# keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
datasets_raw = get_data('a', 200, 50, ('042318', '043018', '051018'), scale=0.5)

print_dataset_info(datasets_raw)


def gen_cnn_partial(input_size, n):
    return gen_maskcnn_polished(input_size, n,
                                out_channel=16,  # (try, 8, 16, 32, 48)
                                kernel_size_l1=9,  # (try 5,9,13)
                                kernel_size_l23=3,
                                act_fn='softplus',
                                pooling_ksize=3,  # (try, 1,3,5,7)
                                pooling_type='avg',  # try (avg, max)  # looks that max works well here?
                                num_layer=2,
                                )  # ( try, 2,3)


def do_something(note):
    opt_config_partial = partial(get_maskcnn_v1_opt_config,
                                 scale=0.01,
                                 smoothness=0.00005,
                                 group=0.0,
                                 legacy=False)

    results = []
    for seed in range(3):
        results.append(train_one(gen_cnn_partial, opt_config_partial,
                                 datasets_raw,
                                 key=f'debug/maskcnn_polished/v1_code/{note}/{seed}',
                                 show_every=1000, legacy_corr=True,
                                 model_seed=seed, return_model=False))

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
