"""this file stores some model configs for `maskcnn_polished_with_local_pcn` models

I create this module because this set of hyper parameters need to be accessed from multiple places.
"""
from copy import deepcopy
from typing import Union, Optional
from itertools import chain, zip_longest
from ...submission import utils
from ... import dir_dict
from os.path import join


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
        (3,)
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


def explored_models_20200218():
    """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200218.py"""

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
        (8, 16, 32)
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (2, 3)
    )

    # inherited from _with_local_pcn
    param_iterator_obj.add_pair(
        'kernel_size_l1',
        (9,)
    )

    # try different kernel sizes.
    param_iterator_obj.add_pair(
        'kernel_size_l23',
        (3, 5, 7, 9)
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
        ('cummean',)
    )

    param_iterator_obj.add_pair(
        'ff_1st_block',
        (True,)
    )

    param_iterator_obj.add_pair(
        'ff_1st_bn_before_act',
        (True, False)
    )

    return param_iterator_obj


def explored_models_20200430():
    """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""

    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560),
    )

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
        (8, 16, 32)
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (2, 3)
    )

    # inherited from _with_local_pcn
    param_iterator_obj.add_pair(
        'kernel_size_l1',
        (9,)
    )

    # try different kernel sizes.
    param_iterator_obj.add_pair(
        'kernel_size_l23',
        (3,)
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
        ('cummean',)
    )

    param_iterator_obj.add_pair(
        'ff_1st_block',
        (True,)
    )

    param_iterator_obj.add_pair(
        'ff_1st_bn_before_act',
        (True, False)
    )

    return param_iterator_obj


def explored_models_20200502():
    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        # 1,2,3
        'seq_length', range(1, 4)
    )

    param_iterator_obj.add_pair(
        'model_prefix', (
            'maskcnn_polished_with_rcnn_k_bl_per_trial',
        )
    )

    param_iterator_obj.add_pair(
        'split_seed',
        # also try some other splits, with each class represented equally.
        ('legacy',),
    )

    param_iterator_obj.add_pair(
        'model_seed',
        # range(5),
        range(1),  # otherwise too long.
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
        (
            # 8,
            16,
            # 32,
        )
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (2, 3)
    )

    # inherited from _with_local_pcn
    param_iterator_obj.add_pair(
        'kernel_size_l1',
        (9,)
    )

    # try different kernel sizes.
    param_iterator_obj.add_pair(
        'kernel_size_l23',
        (3,)
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
        ('cummean',)
    )

    param_iterator_obj.add_pair(
        'ff_1st_block',
        (True,)
    )

    param_iterator_obj.add_pair(
        'ff_1st_bn_before_act',
        (True, False)
    )

    return param_iterator_obj


def explored_models_20200509_cb19():
    """those in scripts/training/cb19/maskcnn_polished_with_rcnn_k_bl/submit_20200509.py"""

    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'dataset_prefix', (
            'cb19',
        )
    )

    param_iterator_obj.add_pair(
        'train_keep',
        (1160, 2320, 4640),
    )

    param_iterator_obj.add_pair(
        'split_seed',
        # also try some other splits, with each class represented equally.
        range(1, )
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
        (40,
         # 100,  # should also try 100 later
         )
    )

    param_iterator_obj.add_pair(
        'out_channel',
        (8, 16)
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (2, 3)
    )

    # inherited from _with_local_pcn
    param_iterator_obj.add_pair(
        'kernel_size_l1',
        (9,)
    )

    # try different kernel sizes.
    param_iterator_obj.add_pair(
        'kernel_size_l23',
        (3,)
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
        ('cummean',)
    )

    param_iterator_obj.add_pair(
        'ff_1st_block',
        (True,)
    )

    param_iterator_obj.add_pair(
        'ff_1st_bn_before_act',
        (True, False)
    )

    return param_iterator_obj


def explored_models_20200514_cb19():
    """those in scripts/training/cb19/maskcnn_polished_with_rcnn_k_bl/submit_20200514.py"""
    param_iterator_obj = explored_models_20200509_cb19()
    param_iterator_obj.add_pair(
        'input_size',
        (50,
         ),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'px_kept',
        (100,
         ),
    )

    return param_iterator_obj


def explored_models_20200515_cb19():
    """those in scripts/training/cb19/maskcnn_polished_with_rcnn_k_bl/submit_20200515.py"""
    param_iterator_obj = explored_models_20200514_cb19()

    param_iterator_obj.add_pair(
        'px_kept',
        (80,
         ),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200515():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        ('quarter', 'half', 'full'),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'model_prefix', (
            'maskcnn_polished_with_rcnn_k_bl_with_cb19',
        )
    )

    param_iterator_obj.add_pair(
        'cb19_px_kept',
        (80, 100),
    )

    param_iterator_obj.add_pair(
        'cb19_split_seed',
        range(1),
    )

    return param_iterator_obj


def explored_models_20200517():
    # explore even smaller data set size
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (320, 640),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200530():
    # explore acc type = 'last', using only last one, to compare with local PCN.
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        ('last',),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200731():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        ('instant',),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200617():
    # explore acc type = 'last', using only last one, to compare with local PCN.
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        ('last',),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200530_2():
    # explore deep FF models with
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (1,),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (4, 5, 6,),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'out_channel',
        # 2,4 for matching parameters,
        # 8, 16, 32 for comparison with recurrent models (with tied weights)
        (2, 4, 8, 16, 32),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200520():
    # explore even smaller data set size
    param_iterator_obj = explored_models_20200430()
    # None means full set.
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200616():
    # explore even smaller data set size
    param_iterator_obj = explored_models_20200430()
    # None means full set.
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (8, 9, 10),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200704():
    # explore acc type = 'last', using only last one, to compare with local PCN.
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        # this will make eval and train match.
        # only using the mean of all average.
        ('cummean_last',),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200705():
    # explore acc type = 'last', using only last one, to compare with local PCN.
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        # this will make eval and train match.
        # only using the mean of all average.
        ('cummean_last',),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200704_2():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('avg',),
    )

    return param_iterator_obj


def explored_models_20200707():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200708():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200709():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (8, 9, 10),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200725_cm_avg():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        range(1, 11),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200725_cm_last():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        range(1, 8),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        # this will make eval and train match.
        # only using the mean of all average.
        ('cummean_last',),
        replace=True,
    )

    return param_iterator_obj


explored_models_20200725_deep_ff = explored_models_20200530_2


def explored_models_20200725_generator(with_source=False):
    # combine all three above, and having consistent number of parameters

    for src, param_this in chain(
            zip_longest(['cm-avg'], explored_models_20200725_cm_avg().generate(), fillvalue='cm-avg'),
            zip_longest(['cm-last'], explored_models_20200725_cm_last().generate(), fillvalue='cm-last'),
            zip_longest(['deep-ff'], explored_models_20200725_deep_ff().generate(), fillvalue='deep-ff'),
    ):
        param_this_ret = {
            'dataset_prefix': 'yuanyuan_8k_a_3day',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        param_this_ret.update(param_this)
        assert param_this_ret['train_keep'] in {None, 2560, 1280}
        # print(len(param_this_ret))
        assert len(param_this_ret) == 26
        if not with_source:
            yield param_this_ret
        else:
            yield src, param_this_ret


def explored_models_20200706():
    param_iterator_obj = explored_models_20200704_2()
    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200516_gaya():
    param_iterator_obj = explored_models_20200430()
    param_iterator_obj.add_pair(
        'train_keep',
        (1900 // 2, 1900, 3800),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'dataset_prefix', (
            'gaya',
        )
    )

    param_iterator_obj.add_pair(
        'input_size',
        (63,),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200519_gaya():
    param_iterator_obj = explored_models_20200516_gaya()

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200711_gaya():
    param_iterator_obj = explored_models_20200516_gaya()

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (1, 2, 3, 4, 5, 6, 7),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200712_gaya():
    param_iterator_obj = explored_models_20200516_gaya()

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (1, 2, 3, 4, 5, 6, 7),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'rcnn_acc_type',
        # this will make eval and train match.
        # only using the mean of all average.
        ('cummean_last',),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200624_gaya():
    param_iterator_obj = explored_models_20200516_gaya()

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (1,),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'num_layer',
        (4, 5, 6,),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'out_channel',
        # 2,4 for matching parameters,
        # 8, 16, 32 for comparison with recurrent models (with tied weights)
        (2, 4, 8, 16, 32),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200523_gaya_feature_extraction_generator():
    # concatenate explored_models_20200516_gaya and explored_models_20200519_gaya
    return chain(
        explored_models_20200516_gaya().generate(),
        explored_models_20200519_gaya().generate()
    )


def explored_models_20200523_8k_feature_extraction_generator():
    for x in chain(
            explored_models_20200218().generate(),
            explored_models_20200430().generate(),
            explored_models_20200520().generate(),
    ):
        if 'train_keep' not in x:
            x['train_keep'] = None
        assert len(x) == 23
        if x['kernel_size_l23'] == 3:
            yield x


def explored_models_20200617_8k_feature_extraction_generator():
    for x in chain(
            explored_models_20200616().generate(),
    ):
        assert len(x) == 23
        assert x['kernel_size_l23'] == 3
        yield x


def explored_models_20200619_8k_feature_extraction_generator():
    for x in chain(
            explored_models_20200617().generate(),
    ):
        assert len(x) == 23
        assert x['kernel_size_l23'] == 3
        assert x['rcnn_acc_type'] == 'last'
        yield x


def explored_models_20200518_gaya():
    param_iterator_obj = explored_models_20200516_gaya()
    param_iterator_obj.add_pair(
        'train_keep',
        # 1900/8 != 1900//8. but it should be fine.
        # 256 is the smallest we can take, because that's batch size.
        (max(1900 // 8, 256), 1900 // 4),
        replace=True,
    )

    return param_iterator_obj


def encode_transfer_learning_cb19_params(params: dict):
    params = deepcopy(params)
    cb19_px_kept = params.pop('cb19_px_kept')
    cb19_split_seed = params.pop('cb19_split_seed')

    params['additional_key'] = f'cb19_px{cb19_px_kept}_ss{cb19_split_seed}'

    return params


def decode_transfer_learning_cb19_params(additional_key):
    s1, s2, s3 = additional_key.split('_')
    assert s1 == 'cb19'
    assert s2.startswith('px')
    assert s3.startswith('ss')

    return {
        'cb19_px_kept': int(s2[2:]),
        'cb19_split_seed': int(s3[2:]),
    }


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

           # please be consistent with values set in thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder.gen_maskcnn_polished_with_rcnn_k_bl  # noqa: E501
           # I can also go super fancy, with `inspect.Signature`
           # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
           # but let's not waste time on such things for now.

           ff_1st_block: bool = False,
           ff_1st_bn_before_act: bool = True,
           kernel_size_l23: int = 3,
           train_keep: Optional[Union[int, str]] = None,
           seq_length: Optional[int] = None,
           px_kept: Optional[int] = None,

           additional_key: Optional[str] = None,
           yhat_reduce_pick: int = -1,
           ):
    if ff_1st_block:
        # then add another two blocks
        additional_list = [
            f'ff1st_{ff_1st_block}',
            f'ff1stbba_{ff_1st_bn_before_act}',
        ]
    else:
        additional_list = []

    if kernel_size_l23 == 3:
        additional_list += []
    else:
        additional_list += [f'k_l23{kernel_size_l23}']

    if train_keep is None:
        additional_list += []
    else:
        additional_list += [f'train_keep_{train_keep}']

    if seq_length is None:
        additional_list += []
    else:
        additional_list += [f'seql_{seq_length}']

    if px_kept is None:
        additional_list += []
    else:
        additional_list += [f'px_{px_kept}']

    if yhat_reduce_pick == -1:
        additional_list += []
    else:
        additional_list += [f'rp_{yhat_reduce_pick}']

    if additional_key is None:
        additional_list += []
    else:
        additional_list += [f'addkey_{additional_key}']

    # suffix itself can contain /
    ret = '/'.join(
        [
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
            f'r_acc{rcnn_acc_type}'
        ]
        + additional_list +
        [
            # regularization, optimization stuff
            f'sc{scale_name}',
            f'sm{smoothness_name}',
            f'l{loss_type}',
            # seed
            f'm_se{model_seed}',
        ]
    )
    # print(ret)
    if train_keep is None:
        added_param_size = 0
    else:
        added_param_size = 1

    if seq_length is None:
        added_param_size += 0
    else:
        added_param_size += 1

    if px_kept is None:
        added_param_size += 0
    else:
        added_param_size += 1

    if additional_key is None:
        added_param_size += 0
    else:
        added_param_size += 1

    if yhat_reduce_pick == -1:
        added_param_size += 0
    else:
        added_param_size += 1

    if not ff_1st_block and kernel_size_l23 == 3:
        assert len(ret.split('/')) == 19 + added_param_size
    elif not ff_1st_block and kernel_size_l23 != 3:
        # never tested.
        raise NotImplementedError
    elif ff_1st_block and kernel_size_l23 == 3:
        assert len(ret.split('/')) == 21 + added_param_size
    elif ff_1st_block and kernel_size_l23 != 3:
        assert len(ret.split('/')) == 22 + added_param_size
    else:
        raise RuntimeError

    return ret


def gen_feature_extraction_global_vars(*, key):
    global_vars = {
        'feature_file_dir': join(
            # for cnbc cluster, whose `/user_data/yimengzh` is not big enough.
            '/home/yimengzh/thesis-v2-large-files',
            # dir_dict['features'],
            'maskcnn_polished_with_rcnn_k_bl',
            key
        ),
        'augment_config': {
            'module_names': ['blstack', 'final_act'],
            'name_mapping': {
                'moduledict.bl_stack': 'blstack',
                'moduledict.final_act': 'final_act',
            }
        }
    }

    return global_vars


def script_keygen(**kwargs):
    # remove scale and smoothness
    del kwargs['scale']
    del kwargs['smoothness']

    key = keygen(**kwargs)

    # remove yuanyuan_8k_a_3day/maskcnn_polished part
    return '+'.join(key.split('/')[2:])
