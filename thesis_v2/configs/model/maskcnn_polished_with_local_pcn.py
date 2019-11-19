"""this file stores some model configs for `maskcnn_polished_with_local_pcn` models

I create this module because this set of hyper parameters need to be accessed from multiple places.
"""
from typing import Union
from ...submission import utils


def explored_models_summer_2019_certain():
    """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn/submit_certain_configs.py"""

    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'split_seed',
        # also try some other splits, with each class represented equally.
        ('legacy',),
    )

    param_iterator_obj.add_pair(
        'model_seed',
        # range(5),
        range(3),  # otherwise too long.
    )

    param_iterator_obj.add_pair(
        'act_fn',
        # should try relu later
        ('relu', 'softplus'),
    )

    param_iterator_obj.add_pair(
        'loss_type',
        ('mse', 'poisson')  # should try mse later
    )

    param_iterator_obj.add_pair(
        'input_size',
        (50, 100)  # should also try 100 later
    )

    param_iterator_obj.add_pair(
        'out_channel',
        (16,)
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (2,)
    )

    param_iterator_obj.add_pair(
        'kernel_size_l1',
        (9,)
    )

    param_iterator_obj.add_pair(
        'pooling_ksize',
        (3,)
    )

    param_iterator_obj.add_pair(
        'pooling_type',
        ('avg',)
    )

    param_iterator_obj.add_pair(
        'bn_before_act',
        (True, False,)  # should try False later
    )

    param_iterator_obj.add_pair(
        'bn_after_fc',
        (False,)  # should try True later
    )

    param_iterator_obj.add_pair(
        ('scale_name', 'scale'),
        lambda: {
            # key is the name, value is the actual value to be passed in as is.
            '0.01': '0.01',
            # '0.001': '0.001',
            # '0.1': '0.1',
        }.items(),
        late_call=True,
    )

    param_iterator_obj.add_pair(
        ('smoothness_name', 'smoothness'),
        lambda: {
            '0.000005': '0.000005',
            # '0.00005': '0.00005',
            # '0.0005': '0.0005',
            # '0.005': '0.005',
        }.items(),
        late_call=True,
    )

    # pcn_bn={pcn_bn}
    param_iterator_obj.add_pair(
        'pcn_bn',
        (True, False,),
    )

    # pcn_bn_post={pcn_bn_post},
    param_iterator_obj.add_pair(
        'pcn_bn_post',
        (False, True,),
    )

    # pcn_bypass={pcn_bypass},
    param_iterator_obj.add_pair(
        'pcn_bypass',
        (False,),
    )

    # pcn_cls={pcn_cls},
    param_iterator_obj.add_pair(
        'pcn_cls',
        range(6),
    )

    # pcn_final_act={pcn_final_act},
    param_iterator_obj.add_pair(
        'pcn_final_act',
        (True, False,)
    )
    # pcn_no_act={pcn_no_act},

    param_iterator_obj.add_pair(
        'pcn_no_act',
        (False,)
    )

    # pcn_bias={pcn_bias},

    param_iterator_obj.add_pair(
        'pcn_bias',
        (True,)
    )

    return param_iterator_obj


def keygen(*,
           split_seed: Union[int, str],
           model_seed: int,
           act_fn: str,
           loss_type: str,
           input_size: int,
           out_channel: int,
           num_layer: int,
           kernel_size_l1: int,
           pooling_ksize: int,
           scale_name: str,
           smoothness_name=str,
           pooling_type: str,
           bn_before_act: bool,
           bn_after_fc: bool,

           pcn_cls: int,
           pcn_bypass: bool,
           pcn_no_act: bool,
           pcn_bn_post: bool,
           pcn_final_act: bool,
           pcn_bn: bool,
           pcn_bias: bool,
           dataset_prefix: str = 'yuanyuan_8k_a_3day',
           model_prefix: str = 'maskcnn_polished_with_local_pcn',
           ):
    # suffix itself can contain /
    return '/'.join([
        f'{dataset_prefix}/{model_prefix}',
        f's_se{split_seed}',
        f'in_sz{input_size}',
        f'out_ch{out_channel}',
        f'num_l{num_layer}',
        f'k_l1{kernel_size_l1}',
        f'k_p{pooling_ksize}',
        f'pt{pooling_type}',
        f'bn_b_act{bn_before_act}',
        f'bn_a_fc{bn_after_fc}',
        f'act{act_fn}',

        # add those local pcn specific stuff
        f'p_c{pcn_cls}',
        f'p_bypass{pcn_bypass}',
        f'p_n_act{pcn_no_act}',
        f'p_bn_p{pcn_bn_post}',
        f'p_act{pcn_final_act}',
        f'p_bn{pcn_bn}',
        f'p_bias{pcn_bias}',

        f'sc{scale_name}',
        f'sm{smoothness_name}',
        f'l{loss_type}',
        f'm_se{model_seed}'
    ])


def script_keygen(**kwargs):
    # remove scale and smoothness
    del kwargs['scale']
    del kwargs['smoothness']
    key = keygen(**kwargs)

    # remove yuanyuan_8k_a_3day/maskcnn_polished part
    return '+'.join(key.split('/')[2:])
