# based on `scripts/debug/feature_approximation/local_pcn_original_imagenet/debug.ipynb`
from functools import partial

import numpy as np
import h5py
from thesis_v2.models.feature_approximation.builder import (
    gen_local_pcn_recurrent_feature_approximator
)

from thesis_v2.training_extra.feature_approximation.opt import get_feature_approximation_opt_config
from thesis_v2.training_extra.feature_approximation.training import train_one
from thesis_v2.configs.model.feature_approximation import keygen, consts

from thesis_v2.models.pcn_local.feature_extraction import get_one_network_meta

meta = get_one_network_meta('PredNetBpE_3CLS')['module_names']


# utils
def get_layer_idx(friendly_name):
    # friendly_name can be
    # convX.in
    # convX.init
    # convX.loop
    # X in 0~9.
    return meta.index(friendly_name)


def fetch_data(grp_name, conv_idx):
    assert conv_idx in range(11)  # 0 through 10.
    feature_file = consts['local_pcn_original_imagenet_imagenet_val']['feature_file']
    slice_to_check = slice(None)
    with h5py.File(feature_file, 'r') as f_feature:
        grp = f_feature[grp_name]
        assert str(get_layer_idx(f'conv{conv_idx}.init')) + '.0' in grp
        num_bottom_up = 1 + len([x for x in grp if x.startswith(str(get_layer_idx(f'conv{conv_idx}.loop')) + '.')])
        # should have at least two bottom up.
        assert num_bottom_up > 1
        # for our current use case.
        assert num_bottom_up == 4

        pcn_in = grp[str(get_layer_idx(f'conv{conv_idx}.in')) + '.0'][slice_to_check]
        # conv.init + all conv.loop
        # num_bottom_up - 1 because of conv.init
        pcn_out_list = [grp[str(get_layer_idx(f'conv{conv_idx}.init')) + '.0'][slice_to_check]] + [
            grp[str(get_layer_idx(f'conv{conv_idx}.loop')) + f'.{x}'][slice_to_check] for x in range(num_bottom_up - 1)]

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
           dataset_prefix,
           basemodel_idx,
           sep,
           ):
    key = keygen(
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        kernel_size=kernel_size,
        bn_pre=bn_pre,
        # conv_idx gives base model index,
        basemodel_idx=basemodel_idx,
        dataset_prefix=dataset_prefix,
        model_prefix=consts[f'local_pcn_original_imagenet_sep{sep}_model_prefix'],
    )
    print(key)

    _dummy, extraction_dataset_name, extraction_setting_name = dataset_prefix.split('+')
    assert _dummy == consts['local_pcn_original_imagenet_imagenet_val']['dataset_prefix_prefix']

    # load and prepare data
    model_name = consts['local_pcn_original_imagenet_imagenet_val']['model_name']
    dataset_this = prepare_data(fetch_data(f'{extraction_dataset_name}/{model_name}/{extraction_setting_name}', basemodel_idx), sep)

    def gen_cnn_partial(in_shape, in_y_shape):
        assert len(in_shape) == 3

        assert len(in_y_shape) == 3
        in_higher_c = in_y_shape[0]
        in_lower_c = in_shape[0] - in_higher_c
        assert in_lower_c > 0
        return gen_local_pcn_recurrent_feature_approximator(
            in_shape_lower=[in_lower_c, in_shape[1], in_shape[2]],
            in_shape_higher=[in_higher_c, in_shape[1], in_shape[2]],
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
              show_every=100,
              model_seed=model_seed,
              max_epoch=7000,
              return_model=False,
              batch_size=32,
              # for conv_idx=0, which is too slow to finish 3 phases.
              # num_phase=1,
              )
