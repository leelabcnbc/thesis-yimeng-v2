"""this file stores some model configs for `maskcnn_polished_with_local_pcn` models

I create this module because this set of hyper parameters need to be accessed from multiple places.
"""
from typing import Union, Optional
from ...submission import utils


def explored_models_20200208():
    """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200208.py"""

    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'split_seed',
        # also try some other splits, with each class represented equally.
        ('legacy',),
    )

    param_iterator_obj.add_pair(
        'model_seed',
        # range(5),
        range(2),  # otherwise too long.
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
        (50,
         # 100,  # should also try 100 later
         )
    )

    param_iterator_obj.add_pair(
        'out_channel',
        (16, 48)
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (2, 3,)
    )

    param_iterator_obj.add_pair(
        'kernel_size_l1',
        (7, 9, 13)
    )

    param_iterator_obj.add_pair(
        'pooling_ksize',
        (3, )
    )

    param_iterator_obj.add_pair(
        'pooling_type',
        ('avg',)
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
            '0.1': '0.1',
        }.items(),
        late_call=True,
    )

    param_iterator_obj.add_pair(
        ('smoothness_name', 'smoothness'),
        lambda: {
            '0.000005': '0.000005',
            # '0.00005': '0.00005',
            # '0.0005': '0.0005',
            '0.005': '0.005',
        }.items(),
        late_call=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        range(1, 5),
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_psize',
        (1,)
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_ptype',
        (None,)
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        ('cummean', 'instant',)
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

           bn_after_fc: bool,

           rcnn_bl_cls: int,
           rcnn_bl_psize: int,
           rcnn_bl_ptype: Optional[str],
           rcnn_acc_type: str,
           dataset_prefix: str = 'yuanyuan_8k_a_3day',
           model_prefix: str = 'maskcnn_polished_with_rcnn_k_bl',
           ):
    # suffix itself can contain /
    ret = '/'.join([
        # model name, dataset name
        f'{dataset_prefix}/{model_prefix}',
        # seed
        f's_se{split_seed}',
        # model arch stuff
        f'in_sz{input_size}',
        f'out_ch{out_channel}',
        f'num_l{num_layer}',
        f'k_l1{kernel_size_l1}',
        f'k_p{pooling_ksize}',
        f'pt{pooling_type}',
        f'bn_a_fc{bn_after_fc}',
        f'act{act_fn}',

        # add those RCNN-BL specific model arch stuff
        f'r_c{rcnn_bl_cls}',
        f'r_psize{rcnn_bl_psize}',
        f'r_ptype{rcnn_bl_ptype}',
        f'r_acc{rcnn_acc_type}',

        # regularization, optimization stuff
        f'sc{scale_name}',
        f'sm{smoothness_name}',
        f'l{loss_type}',
        # seed
        f'm_se{model_seed}',
    ])
    print(ret)
    assert len(ret.split('/')) == 19

    return ret


def script_keygen(**kwargs):
    # remove scale and smoothness
    del kwargs['scale']
    del kwargs['smoothness']
    key = keygen(**kwargs)

    # remove yuanyuan_8k_a_3day/maskcnn_polished part
    return '+'.join(key.split('/')[2:])
