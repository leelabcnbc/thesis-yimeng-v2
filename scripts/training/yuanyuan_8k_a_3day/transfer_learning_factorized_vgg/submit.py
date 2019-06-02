from os.path import dirname, relpath, realpath
from itertools import product

import h5py

from thesis_v2 import dir_dict, join
from thesis_v2.submission import utils, transfer_learing

from key_utils import script_keygen

# this is used to find master.py
# you can't drop relpath; this file runs in host, but call_script runs in
# singularity.
# print(dir_dict)
current_dir = realpath(dirname(__file__))
# print(current_dir, dir_dict['root'])
current_dir = relpath(current_dir, dir_dict['root'])
# print(current_dir)

call_script = """
from sys import path
from thesis_v2 import dir_dict, join
path.insert(0, join(dir_dict['root'], {current_dir}))
from master import master

master(
split_seed={split_seed},
sparse={sparse},
model_seed={model_seed},
act_fn={act_fn},
loss_type={loss_type},
suffix={suffix},
)
""".strip()

split_seed_list = ('legacy',)
model_seed_list = (0,)
# this is enough. I'm more concerned with getting the order of magnitude
# right, not state of the art performance.
sparse_list = ('0.1', '0.01', '0.001', '0.0001', '0.00001')
act_fn_list = ('softplus',)
loss_type_list = ('poisson',)

layers_to_check = {
    'vgg16': {'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
              },
    'vgg16_bn': {'pool1',
                 'conv2_1', 'conv2_2', 'pool2',
                 'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
                 },
    'vgg11': {'pool1',
              'conv2_1', 'pool2',
              'conv3_1', 'conv3_2', 'pool3',
              },
    'vgg11_bn': {'pool1',
                 'conv2_1', 'pool2',
                 'conv3_1', 'conv3_2', 'pool3',
                 },
}


# open that file, and iterate to get all cases to process.
def get_all_suffix():
    file_to_save_feature = join(
        dir_dict['features'],
        'cnn_feature_extraction',
        'yuanyuan_8k_a',
        'vgg.hdf5'
    )

    all_cases = []

    def callback(name, obj):
        if isinstance(obj, h5py.Dataset):
            meta = transfer_learing.fetch_meta(name, obj)

            if meta['setting'] == 'quarter':
                if meta['layer_name'] in layers_to_check.get(meta['network'],
                                                             set()):
                    all_cases.append('/'.join(meta['splitted_name'][1:]))

    with h5py.File(file_to_save_feature, 'r') as f_feature:
        f_feature.visititems(callback)

    assert len(set(all_cases)) == len(all_cases)
    print(all_cases)
    return all_cases


def param_iterator(*, include_sparse=True):
    if include_sparse:
        for suffix, split_seed, model_seed, sparse, act_fn, loss_type in product(
                get_all_suffix(), split_seed_list,
                model_seed_list, sparse_list, act_fn_list,
                loss_type_list
        ):
            yield {
                'suffix': suffix,
                'split_seed': split_seed,
                'sparse': sparse,
                'model_seed': model_seed,
                'act_fn': act_fn,
                'loss_type': loss_type,
            }
    else:
        for suffix, split_seed, model_seed, act_fn, loss_type in product(
                get_all_suffix(), split_seed_list,
                model_seed_list, act_fn_list,
                loss_type_list
        ):
            yield {
                'suffix': suffix,
                'split_seed': split_seed,
                'model_seed': model_seed,
                'act_fn': act_fn,
                'loss_type': loss_type,
            }


def main():
    script_dict = dict()
    for param_dict in param_iterator():
        key_this = script_keygen(**param_dict)

        script_dict[key_this] = utils.call_script_formatter(
            call_script, set(),
            current_dir=current_dir,
            **param_dict,
        )

    utils.submit(
        script_dict, 'transfer_learning', 'standard', True,
        dirname_relative='scripts+yuanyuan_8k_a_3day+transfer_learning_factorized_vgg'  # noqa: E501
    )


if __name__ == '__main__':
    main()
