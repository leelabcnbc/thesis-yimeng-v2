from os.path import dirname, relpath, realpath

import h5py

from thesis_v2 import dir_dict, join
from thesis_v2.submission import utils
from thesis_v2.training_extra.transfer_learning.meta import fetch_meta

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
shuffle_type={shuffle_type},
split_seed={split_seed},
sparse={sparse},
model_seed={model_seed},
act_fn={act_fn},
loss_type={loss_type},
suffix={suffix},
)
""".strip()

shuffle_type_list = ('legacy', 'paired')
split_seed_list = range(3)
model_seed_list = (0,)
# this is enough. I'm more concerned with getting the order of magnitude
# right, not state of the art performance.
sparse_list = ('0.1', '0.01', '0.001', '0.0001', '0.00001')
act_fn_list = ('softplus', 'relu')
loss_type_list = ('poisson', 'mse')

layers_to_check = {
    'vgg16': {
        'pool1',
        'conv2_1', 'conv2_2', 'pool2',
        'conv3_1', 'conv3_2', 'conv3_3',
        'pool3',
    },
    'vgg16_bn': {
        'pool1',
        'conv2_1', 'conv2_2', 'pool2',
        'conv3_1', 'conv3_2', 'conv3_3',
        'pool3',
    },
    'vgg11': {
        'pool1',
        'conv2_1', 'pool2',
        'conv3_1', 'conv3_2',
        'pool3',
    },
    'vgg11_bn': {
        'pool1',
        'conv2_1', 'pool2',
        'conv3_1', 'conv3_2',
        'pool3',
    },
}

feature_file_name = join(
    dir_dict['features'],
    'cnn_feature_extraction',
    'crcns_pvc8',
    'vgg.hdf5'
)


# open that file, and iterate to get all cases to process.
def get_all_suffix():
    all_cases = []

    def callback(name, obj):
        if isinstance(obj, h5py.Dataset):
            meta = fetch_meta(obj, name)
            if meta['dataset'] == 'large':
                if meta['layer_name'] in layers_to_check.get(meta['network'],
                                                             set()):
                    all_cases.append('/'.join(meta['splitted_name'][1:]))

    with h5py.File(feature_file_name, 'r') as f_feature:
        f_feature.visititems(callback)

    assert len(set(all_cases)) == len(all_cases)
    print(all_cases)
    return all_cases


param_iterator_obj = utils.ParamIterator()

param_iterator_obj.add_pair(
    'suffix', get_all_suffix, late_call=True,
)

param_iterator_obj.add_pair(
    'shuffle_type', shuffle_type_list,
)

param_iterator_obj.add_pair(
    'split_seed', split_seed_list,
)

param_iterator_obj.add_pair(
    'sparse', sparse_list,
)

param_iterator_obj.add_pair(
    'model_seed', model_seed_list,
)

param_iterator_obj.add_pair(
    'act_fn', act_fn_list,
)

param_iterator_obj.add_pair(
    'loss_type', loss_type_list,
)


def param_iterator(*, include_sparse=True):
    if include_sparse:
        return param_iterator_obj.generate()
    else:
        return param_iterator_obj.generate(lambda x: x != 'sparse')


def main():
    from key_utils import script_keygen

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
        dirname_relative='scripts+crcns_pvc8_large+transfer_learning_factorized_vgg'  # noqa: E501
    )


if __name__ == '__main__':
    main()
