# based on `scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn_feature_approximation/debug.ipynb`
from functools import partial

import numpy as np
import h5py
from os.path import join
from thesis_v2.models.feature_approximation.builder import (
    gen_local_pcn_recurrent_feature_approximator
)

from thesis_v2.training_extra.feature_approximation.opt import get_feature_approximation_opt_config
from thesis_v2.training_extra.feature_approximation.training import train_one
from thesis_v2.configs.model.feature_approximation import keygen, consts

global_vars = consts['local_pcn_feature_extraction_certain_configs_yuanyuan_8k_a']


def get_layer_idx(friendly_name):
    return global_vars['augment_config']['module_names'].index(friendly_name)


def fetch_data(key_script, grp_name):
    feature_file = join(global_vars['feature_file_dir'], key_script + '.hdf5')
    print('file', feature_file)
    slice_to_check = slice(None)
    with h5py.File(feature_file, 'r') as f_feature:
        grp = f_feature[grp_name]
        num_bottom_up = len([x for x in grp if x.startswith(str(get_layer_idx('bottomup')) + '.')])
        assert num_bottom_up > 2

        pcn_in = grp[str(get_layer_idx('topdown')) + '.0'][slice_to_check]
        pcn_out_list = [grp[str(get_layer_idx('bottomup')) + f'.{x}'][slice_to_check] for x in range(num_bottom_up - 1)]

    print((pcn_in.shape, pcn_in.mean(), pcn_in.std(), pcn_in.min(), pcn_in.max()))
    print([(x.shape, x.mean(), x.std(), x.min(), x.max()) for x in pcn_out_list])

    return {
        'in': pcn_in,
        'out_list': pcn_out_list,
    }


def prepare_data(data_dict, sep):
    num_out = len(data_dict['out_list'])
    x_train = []
    y_train = []
    assert 0 < sep < num_out
    for idx1 in range(num_out):
        for idx2 in range(idx1 + 1, num_out):
            if idx2 - idx1 != sep:
                continue
            print((idx1, idx2))
            x_train.append(np.concatenate([data_dict['in'], data_dict['out_list'][idx1]], axis=1))
            y_train.append(data_dict['out_list'][idx2] - data_dict['out_list'][idx1])

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    print((x_train.shape, y_train.shape))

    dataset_this = {
        'X_train': x_train,
        'y_train': y_train,
    }

    return dataset_this


def master(*,
           model_seed,
           act_fn,
           loss_type,
           kernel_size,
           bn_pre,
           basemodel_key_script,
           basemodel_idx,
           sep,
           ):
    key = keygen(
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        kernel_size=kernel_size,
        bn_pre=bn_pre,
        basemodel_idx=basemodel_idx,
        model_prefix=consts[f'local_pcn_recurrent_sep{sep}_model_prefix']
    )
    print(key)

    # load and prepare data
    dataset_this = prepare_data(fetch_data(basemodel_key_script, 'X_train'), sep)

    def gen_cnn_partial(in_shape):
        # I assume two inputs have the same number of channels and shapes.
        assert len(in_shape) == 3
        assert in_shape[0] % 2 == 0
        return gen_local_pcn_recurrent_feature_approximator(
            in_shape_lower=[in_shape[0]//2, in_shape[1], in_shape[2]],
            in_shape_higher=[in_shape[0]//2, in_shape[1], in_shape[2]],
            kernel_size=kernel_size,
            act_fn=act_fn,
            batchnorm_pre=bn_pre,
        )

    opt_config_partial = partial(get_feature_approximation_opt_config,
                                 loss_type=loss_type)

    train_one(arch_json_partial=gen_cnn_partial,
              opt_config_partial=opt_config_partial,
              datasets=dataset_this,
              key=key,
              show_every=1000,
              model_seed=model_seed,
              max_epoch=5000,
              return_model=False)
