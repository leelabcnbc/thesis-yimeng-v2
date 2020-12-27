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


def explored_models_20200801():
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

    param_iterator_obj.add_pair(
        'rcnn_bl_cls',
        (5, 6, 7),
        replace=True,
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


def explored_models_20201001_generator(with_source=False, largest_cls=None):
    # similar to explored_models_20200725_generator, with more channels.
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

        assert param_this_ret['out_channel'] in {2, 4, 8, 16, 32}
        if param_this_ret['out_channel'] in {2, 4, 8}:
            continue

        if largest_cls is not None and param_this_ret['rcnn_bl_cls'] > largest_cls:
            continue

        param_this_ret['out_channel'] = {16: 48, 32: 64}[param_this_ret['out_channel']]

        # print(len(param_this_ret))
        assert len(param_this_ret) == 26
        if not with_source:
            yield param_this_ret
        else:
            yield src, param_this_ret


def explored_models_20201101_generator(with_source=False, largest_cls=7, act_fn_inner_list=None):
    if act_fn_inner_list is None:
        act_fn_inner_list = ['sigmoid', 'tanh']
    # using sigmoid as internal units
    for src, param_this in chain(
            zip_longest(['cm-avg'], explored_models_20200725_cm_avg().generate(), fillvalue='cm-avg'),
            zip_longest(['cm-last'], explored_models_20200725_cm_last().generate(), fillvalue='cm-last'),
            zip_longest(['deep-ff'], explored_models_20200725_deep_ff().generate(), fillvalue='deep-ff'),
    ):
        for act_fn_inner in act_fn_inner_list:
            param_this_ret = {
                'dataset_prefix': 'yuanyuan_8k_a_3day',
                'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
                'yhat_reduce_pick': -1,
            }
            param_this_ret.update(param_this)
            assert param_this_ret['train_keep'] in {None, 2560, 1280}

            assert param_this_ret['out_channel'] in {2, 4, 8, 16, 32}
            if param_this_ret['out_channel'] in {2, 4}:
                continue

            if largest_cls is not None and param_this_ret['rcnn_bl_cls'] > largest_cls:
                continue

            param_this_ret['act_fn_inner'] = act_fn_inner

            # print(len(param_this_ret))
            assert len(param_this_ret) == 27
            if not with_source:
                yield param_this_ret
            else:
                yield src, param_this_ret


def explored_models_20201012_generator(with_source=False):
    for x in chain(
            explored_models_20200801_generator(with_source=with_source),
            explored_models_20200801_2_generator(with_source=with_source),
    ):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x

        assert param_dict['out_channel'] in {8, 16, 32}
        if param_dict['out_channel'] in {8}:
            continue

        param_dict['out_channel'] = {16: 48, 32: 64}[param_dict['out_channel']]

        assert len(param_dict) == 26
        if not with_source:
            yield param_dict
        else:
            yield src, param_dict


def explored_models_20201114_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True, False)
    # inst-avg, inst-last models, trained using multi path
    for x in chain(
            explored_models_20200801_generator(with_source=with_source),
            explored_models_20200801_2_generator(with_source=with_source),
    ):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x

        if param_dict['rcnn_bl_cls'] == 1:
            continue

        for separate_bn in separate_bn_list:
            param_dict_ret = deepcopy(param_dict)
            param_dict_ret['multi_path'] = True
            param_dict_ret['multi_path_separate_bn'] = separate_bn
            assert len(param_dict_ret) == 28
            if not with_source:
                yield param_dict_ret
            else:
                yield src, param_dict_ret


def explored_models_20201118_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True, False)
    for x in explored_models_20200725_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x

        if param_dict['rcnn_bl_cls'] == 1:
            continue

        for separate_bn in separate_bn_list:
            param_dict_ret = deepcopy(param_dict)
            param_dict_ret['multi_path'] = True
            param_dict_ret['multi_path_separate_bn'] = separate_bn
            assert len(param_dict_ret) == 28
            if not with_source:
                yield param_dict_ret
            else:
                yield src, param_dict_ret


def explored_models_20201215_tang_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True,)
    for src, param_dict in chain(
            explored_models_20201002_tang_generator(with_source=True, additional_keys=('0,500',)),
            explored_models_20201018_tang_generator(with_source=True, additional_keys=('0,500',))
    ):
        if param_dict['rcnn_bl_cls'] == 1:
            continue
        for separate_bn in separate_bn_list:
            param_dict_ret = deepcopy(param_dict)
            param_dict_ret['multi_path'] = True
            param_dict_ret['multi_path_separate_bn'] = separate_bn
            assert len(param_dict_ret) == 29
            if not with_source:
                yield param_dict_ret
            else:
                yield src, param_dict_ret


# 16/32 ch, 2 layer ablation models.
def explored_models_20201218_tang_generator(with_source=False):
    for src, param_dict in explored_models_20201215_tang_generator(
            with_source=True
    ):
        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue

        # only study those good models in `maskcnn_polished_with_rcnn_k_bl/20201118_collect-separatebn.ipynb`
        if not (
                param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 1400}
        ):
            continue

        for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
            for prefix in ['leD', 'geD']:
                param_dict_ret = deepcopy(param_dict)
                assert param_dict_ret['multi_path']
                param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                assert len(param_dict_ret) == 30
                if not with_source:
                    yield param_dict_ret
                else:
                    yield src, param_dict_ret


# 16/32 ch, 2 layer ablation models, only certain length
def explored_models_20201221_tang_generator(with_source=False):
    for src, param_dict in explored_models_20201215_tang_generator(
            with_source=True
    ):
        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue

        # only study those good models in `maskcnn_polished_with_rcnn_k_bl/20201118_collect-separatebn.ipynb`
        if not (
                param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 1400}
        ):
            continue

        for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
            for prefix in ['onlyD', ]:
                param_dict_ret = deepcopy(param_dict)
                assert param_dict_ret['multi_path']
                param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                assert len(param_dict_ret) == 30
                if not with_source:
                    yield param_dict_ret
                else:
                    yield src, param_dict_ret


def explored_models_20201205_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True,)
    for x in explored_models_20200725_generator(with_source=True):
        src, param_dict = x

        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue

        # only study those good models in `maskcnn_polished_with_rcnn_k_bl/20201118_collect-separatebn.ipynb`
        if not (
                src == 'cm-avg' and param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 5120}
        ):
            continue

        for separate_bn in separate_bn_list:
            # then add geDX variants and leDX variants.
            for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
                for prefix in ['leD', 'geD']:
                    param_dict_ret = deepcopy(param_dict)
                    param_dict_ret['multi_path'] = True
                    param_dict_ret['multi_path_separate_bn'] = separate_bn
                    param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                    assert len(param_dict_ret) == 29
                    if not with_source:
                        yield param_dict_ret
                    else:
                        yield src, param_dict_ret


def explored_models_20201205_2_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True,)
    for x in explored_models_20201114_generator(with_source=True, separate_bn_list=separate_bn_list):
        src, param_dict = x
        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue
        # only study those good models in `maskcnn_polished_with_rcnn_k_bl/20201118_collect-separatebn.ipynb`
        if not (
                src == 'inst-avg' and param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 5120}
        ):
            continue

        for separate_bn in separate_bn_list:
            # then add geDX variants and leDX variants.
            for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
                for prefix in ['leD', 'geD']:
                    param_dict_ret = deepcopy(param_dict)
                    param_dict_ret['multi_path'] = True
                    param_dict_ret['multi_path_separate_bn'] = separate_bn
                    param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                    assert len(param_dict_ret) == 29
                    if not with_source:
                        yield param_dict_ret
                    else:
                        yield src, param_dict_ret


def explored_models_20201213_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True,)
    for x in explored_models_20200725_generator(with_source=True):
        src, param_dict = x

        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue

        # only study those good models in `maskcnn_polished_with_rcnn_k_bl/20201118_collect-separatebn.ipynb`
        if not (
                src == 'cm-last' and param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 5120}
        ):
            continue

        for separate_bn in separate_bn_list:
            # then add geDX variants and leDX variants.
            for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
                for prefix in ['leD', 'geD']:
                    param_dict_ret = deepcopy(param_dict)
                    param_dict_ret['multi_path'] = True
                    param_dict_ret['multi_path_separate_bn'] = separate_bn
                    param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                    assert len(param_dict_ret) == 29
                    if not with_source:
                        yield param_dict_ret
                    else:
                        yield src, param_dict_ret


def explored_models_20201213_2_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True,)
    for x in explored_models_20201114_generator(with_source=True, separate_bn_list=separate_bn_list):
        src, param_dict = x
        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue
        if not (
                src == 'inst-last' and param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 5120}
        ):
            continue

        for separate_bn in separate_bn_list:
            # then add geDX variants and leDX variants.
            for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
                for prefix in ['leD', 'geD']:
                    param_dict_ret = deepcopy(param_dict)
                    param_dict_ret['multi_path'] = True
                    param_dict_ret['multi_path_separate_bn'] = separate_bn
                    param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                    assert len(param_dict_ret) == 29
                    if not with_source:
                        yield param_dict_ret
                    else:
                        yield src, param_dict_ret


def explored_models_20201221_generator(with_source=False, separate_bn_list=None):
    if separate_bn_list is None:
        separate_bn_list = (True,)

    for x in chain(
            explored_models_20201114_generator(
                with_source=True, separate_bn_list=separate_bn_list
            ),
            explored_models_20201118_generator(
                with_source=True, separate_bn_list=separate_bn_list
            ),
    ):
        src, param_dict = x
        if param_dict['rcnn_bl_cls'] not in range(2, 7 + 1):
            continue
        if not (
                param_dict['num_layer'] == 2 and
                param_dict['out_channel'] in {16, 32} and
                param_dict['train_keep'] in {None, 5120}
        ):
            continue

        for separate_bn in separate_bn_list:
            # then add geDX variants and leDX variants.
            for depth_this in range(1, param_dict['rcnn_bl_cls'] + 1):
                for prefix in ['onlyD', ]:
                    param_dict_ret = deepcopy(param_dict)
                    param_dict_ret['multi_path'] = True
                    param_dict_ret['multi_path_separate_bn'] = separate_bn
                    param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                    assert len(param_dict_ret) == 29
                    if not with_source:
                        yield param_dict_ret
                    else:
                        yield src, param_dict_ret


def explored_models_20201003_generator(with_source=False):
    # similar to explored_models_20200725_generator, with more channels.
    # combine all three above, and having consistent number of parameters
    for x in explored_models_20200725_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x
        param_dict = deepcopy(param_dict)
        param_dict['blstack_norm_type'] = 'instancenorm'
        assert len(param_dict) == 27
        if not with_source:
            yield param_dict
        else:
            yield src, param_dict


def explored_models_20200801_generator(with_source=False, cnbc_prefix=False):
    if cnbc_prefix:
        extra_key_1 = {
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl.20200731',
        }
        extra_key_2 = {
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl.20200530',
        }
        extra_key_3 = {
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl.20200617',
        }
    else:
        extra_key_1 = None
        extra_key_2 = None
        extra_key_3 = None

    for src, param_this in chain(
            zip_longest(['inst-avg'],
                        explored_models_20200731().generate(extra_keys=extra_key_1), fillvalue='inst-avg'),
            zip_longest(['inst-last'],
                        explored_models_20200530().generate(extra_keys=extra_key_2), fillvalue='inst-last'),
            zip_longest(['inst-last'],
                        explored_models_20200617().generate(extra_keys=extra_key_3), fillvalue='inst-last'),
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


def explored_models_20200801_2_generator(with_source=False):
    for src, param_this in chain(
            zip_longest(['inst-avg'], explored_models_20200801().generate(), fillvalue='inst-avg'),
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


def explored_models_20200802_gaya():
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
        ('instant',),
        replace=True,
    )

    param_iterator_obj.add_pair(
        'yhat_reduce_pick',
        ('none',),
    )

    return param_iterator_obj


def explored_models_20200802_2_gaya():
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
        ('last',),
        replace=True,
    )

    return param_iterator_obj


def explored_models_20200801_gaya_generator(with_source=False):
    # combine all three above, and having consistent number of parameters

    for src, param_this in chain(
            zip_longest(['cm-avg'], explored_models_20200711_gaya().generate(), fillvalue='cm-avg'),
            zip_longest(['cm-last'], explored_models_20200712_gaya().generate(), fillvalue='cm-last'),
            zip_longest(['deep-ff'], explored_models_20200624_gaya().generate(), fillvalue='deep-ff'),
    ):
        param_this_ret = {
            'dataset_prefix': 'gaya',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        param_this_ret.update(param_this)
        assert param_this_ret['train_keep'] in {1900 // 2, 1900, 3800}
        # print(len(param_this_ret))
        assert len(param_this_ret) == 26
        if not with_source:
            yield param_this_ret
        else:
            yield src, param_this_ret


def explored_models_20200819_tang_generator(with_source=False):
    # combine all three above, and having consistent number of parameters
    for x in explored_models_20200801_gaya_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x
        param_dict = deepcopy(param_dict)
        param_dict['dataset_prefix'] = 'tang'
        param_dict['train_keep'] = {3800: 1400, 1900: 700, 1900 // 2: 350}[param_dict['train_keep']]
        assert param_dict['train_keep'] in {350, 700, 1400}
        assert len(param_dict) == 26
        if not with_source:
            yield param_dict
        else:
            yield src, param_dict


def explored_models_20200930_tang_generator(with_source=False):
    # combine all three above, and having consistent number of parameters
    for x in explored_models_20200819_tang_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x
        param_dict = deepcopy(param_dict)
        param_dict['input_size'] = 37
        assert len(param_dict) == 26
        if not with_source:
            yield param_dict
        else:
            yield src, param_dict


def explored_models_20201018_tang_generator(with_source=False, additional_keys=('0,500',)):
    # combine all three above, and having consistent number of parameters
    for x in explored_models_20200914_tang_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x
        param_dict = deepcopy(param_dict)
        param_dict['input_size'] = 37
        for key in additional_keys:
            # default (0,100)
            param_dict['additional_key'] = key
            assert len(param_dict) == 27
            if not with_source:
                yield deepcopy(param_dict)
            else:
                yield src, deepcopy(param_dict)


def explored_models_20201002_tang_generator(with_source=False, additional_keys=('0,500', '400,500')):
    for x in explored_models_20200930_tang_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x

        for key in additional_keys:
            # default (0,100)
            param_dict['additional_key'] = key
            assert len(param_dict) == 27
            if not with_source:
                yield deepcopy(param_dict)
            else:
                yield src, deepcopy(param_dict)


def explored_models_20201026_tang_generator(with_source=False):
    # Use 80-500
    #   Prediction: FF becomes worse
    #   Percentage of improvement will be bigger
    # It could be due to absolute lower performance as well
    # A side point.
    return explored_models_20201002_tang_generator(with_source=with_source, additional_keys=('80,500',))


def explored_models_20200914_tang_generator(with_source=False):
    # combine all three above, and having consistent number of parameters
    for x in explored_models_20200802_gaya_generator(with_source=with_source, contain_model_prefix=True):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x
        param_dict = deepcopy(param_dict)
        param_dict['dataset_prefix'] = 'tang'
        param_dict['train_keep'] = {3800: 1400, 1900: 700, 1900 // 2: 350}[param_dict['train_keep']]
        assert param_dict['train_keep'] in {350, 700, 1400}
        assert len(param_dict) == 26
        if not with_source:
            yield param_dict
        else:
            yield src, param_dict


def explored_models_20200822_tang_generator(with_source=False):
    for x in explored_models_20200819_tang_generator(with_source=with_source):
        if not with_source:
            param_dict = x
            src = None
        else:
            src, param_dict = x

        # use full response, instead of default (0,100)
        param_dict['additional_key'] = '0,500'
        assert len(param_dict) == 27
        if not with_source:
            yield param_dict
        else:
            yield src, param_dict


def explored_models_20200802_gaya_generator(with_source=False, contain_model_prefix=False):
    # combine all three above, and having consistent number of parameters

    for src, param_this in chain(
            zip_longest(['inst-avg'], explored_models_20200802_gaya().generate(), fillvalue='inst-avg'),
            zip_longest(['inst-last'], explored_models_20200802_2_gaya().generate(), fillvalue='inst-last'),
    ):
        param_this_ret = {
            'dataset_prefix': 'gaya',
            'yhat_reduce_pick': -1,
        }
        if contain_model_prefix:
            param_this_ret['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        param_this_ret.update(param_this)
        assert param_this_ret['train_keep'] in {1900 // 2, 1900, 3800}
        # print(len(param_this_ret))
        if contain_model_prefix:
            assert len(param_this_ret) == 26
        else:
            assert len(param_this_ret) == 25
        if not with_source:
            yield param_this_ret
        else:
            yield src, param_this_ret


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
           blstack_norm_type: str = 'batchnorm',
           act_fn_inner: str = 'same',
           multi_path=False,
           multi_path_separate_bn=None,
           multi_path_hack: Optional[str] = None,
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

    if blstack_norm_type == 'batchnorm':
        additional_list += []
    else:
        additional_list += [f'bnt_{blstack_norm_type}']

    if act_fn_inner == 'same':
        additional_list += []
    else:
        additional_list += [f'actin_{act_fn_inner}']

    if not multi_path:
        additional_list += []
    else:
        additional_list += [f'mp_{multi_path}']

    if multi_path_separate_bn is None:
        additional_list += []
    else:
        additional_list += [f'mpb_{multi_path_separate_bn}']

    if multi_path_hack is None:
        additional_list += []
    else:
        additional_list += [f'mph_{multi_path_hack}']

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

    if blstack_norm_type == 'batchnorm':
        added_param_size += 0
    else:
        added_param_size += 1

    if act_fn_inner == 'same':
        added_param_size += 0
    else:
        added_param_size += 1

    if not multi_path:
        added_param_size += 0
    else:
        added_param_size += 1

    if multi_path_separate_bn is None:
        added_param_size += 0
    else:
        added_param_size += 1

    if multi_path_hack is None:
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


def add_common_part_8k(param_iterator_obj):
    param_iterator_obj.add_pair(
        'train_keep',
        (1280, 2560, None),
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
        [('0.01', '0.01')],
    )

    param_iterator_obj.add_pair(
        ('smoothness_name', 'smoothness'),
        [('0.000005', '0.000005')],
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
        'ff_1st_block',
        (True,)
    )

    param_iterator_obj.add_pair(
        'ff_1st_bn_before_act',
        (True, False)
    )


def main_models_8k_generator(with_source):
    # this only contains models
    # presented in the thesis paper.

    def model_r():
        """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""
        param_iterator_obj = utils.ParamIterator()

        add_common_part_8k(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32, 48, 64)
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2, 3)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(1, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # cm-last
                # this is different from (`cummean`, -1).
                # for loss calculation.
                # for (`cummean`, -1),
                # loss used all iterations during training, due to broadcasting.
                # but early stopping only used the last.
                #
                # by definition, we should NOT use all iterations,
                # but only the last.

                ('cummean_last', -1),
                # cm-avg
                ('cummean', 'none'),
                # inst-last
                ('last', -1),
                # inst-avg
                ('instant', 'none'),
            ],
        )

        return param_iterator_obj

    def model_additional_ff():
        param_iterator_obj = utils.ParamIterator()
        add_common_part_8k(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32, 48, 64)
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (4, 5, 6)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(1, 2),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                ('cummean', -1),
            ],
        )

        return param_iterator_obj

    for x in chain(
            model_r().generate(),
            model_additional_ff().generate(),
    ):
        source = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
            (-1, 'cummean'): 'deep-ff',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'yuanyuan_8k_a_3day'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'

        assert len(x) == 26
        if with_source:
            yield source, x
        else:
            yield x


def main_models_8k_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in main_models_8k_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)
    assert len(key_all) == (
        # 480 + # 3,5 layer ff models to compare with R
        # 480 2, 3 layer FF models are actually computed
        # by avergaing multiple R=1 cases.
            480 * 24 +  # corresponding recurrent models
            # redundant r models that are the same as 3-layer FF models.
            # their performance are averaged with ff models
            240 * 4 +
            # 4,5,6 layer FF models.
            # this might be used in appendix.
            720 +
            # redundant r models that are the same as 2-layer FF models.
            # their performance are averaged with ff models
            240 * 4
    )

    # check that scripts specified in the README can indeed cover all
    # the cases.
    key_all_2nd = set()
    key_all_2nd_full = set()
    for y in chain(
            explored_models_20200530().generate(),
            explored_models_20200530_2().generate(),
            explored_models_20200617().generate(),
            explored_models_20200704().generate(),
            explored_models_20200705().generate(),
            explored_models_20200707().generate(),
            explored_models_20200708().generate(),
            explored_models_20200709().generate(),
            explored_models_20200731().generate(),
            explored_models_20200801().generate(),
            explored_models_20201001_generator(),
            explored_models_20201012_generator(),
    ):
        y_full = {
            'dataset_prefix': 'yuanyuan_8k_a_3day',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        y_full.update(y)

        key_y = keygen(
            # skip these two because they are of float
            **{k: v for k, v in y_full.items() if k not in {'scale', 'smoothness'}}
        )

        assert key_y not in key_all_2nd_full
        key_all_2nd_full.add(key_y)

        # remove some extra models.
        if y_full['rcnn_bl_cls'] > 7:
            continue
        if y_full['out_channel'] not in {8, 16, 32, 48, 64}:
            continue

        assert key_y not in key_all_2nd
        key_all_2nd.add(key_y)

    assert key_all_2nd == key_all
    assert key_all_2nd_full >= key_all_2nd


def multipath_models_8k_generator(with_source):
    # 2L, 16/32 ch models,
    # cls 2 through 7
    def model_r():
        """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""
        param_iterator_obj = utils.ParamIterator()

        add_common_part_8k(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32,)
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2, 3,)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(2, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # cm-last
                # this is different from (`cummean`, -1).
                # for loss calculation.
                # for (`cummean`, -1),
                # loss used all iterations during training, due to broadcasting.
                # but early stopping only used the last.
                #
                # by definition, we should NOT use all iterations,
                # but only the last.

                ('cummean_last', -1),
                # cm-avg
                ('cummean', 'none'),
                # inst-last
                ('last', -1),
                # inst-avg
                ('instant', 'none'),
            ],
        )

        return param_iterator_obj

    for x in model_r().generate():
        source = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'yuanyuan_8k_a_3day'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        x['multi_path'] = True
        x['multi_path_separate_bn'] = True

        assert len(x) == 28
        if with_source:
            yield source, x
        else:
            yield x


def multipath_models_8k_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in multipath_models_8k_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)

    # 16 variants per size.
    # 4 readout
    # 6 cls
    # 3 ch
    # 2 layer
    # 3 training size
    assert len(key_all) == 16 * 4 * 6 * 3 * 2 * 3

    # check that scripts specified in the README can indeed cover all
    # the cases.
    key_all_2nd = set()
    key_all_2nd_full = set()
    for y in chain(
            explored_models_20201114_generator(),
            explored_models_20201118_generator(),
    ):
        y_full = {
            'dataset_prefix': 'yuanyuan_8k_a_3day',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        y_full.update(y)
        key_y = keygen(
            # skip these two because they are of float
            **{k: v for k, v in y_full.items() if k not in {'scale', 'smoothness'}}
        )

        assert key_y not in key_all_2nd_full
        key_all_2nd_full.add(key_y)

        # remove some extra models.
        if y_full['rcnn_bl_cls'] > 7:
            continue
        if y_full['out_channel'] not in {8, 16, 32}:
            continue
        if not y_full['multi_path_separate_bn']:
            continue

        assert key_y not in key_all_2nd
        key_all_2nd.add(key_y)

    assert key_all_2nd == key_all
    assert key_all_2nd_full >= key_all_2nd


def add_common_part_ns2250(param_iterator_obj):
    param_iterator_obj.add_pair(
        'train_keep',
        (350, 700, 1400),
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
        (37,
         # 100,  # should also try 100 later
         )
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
        [('0.01', '0.01')],
    )

    param_iterator_obj.add_pair(
        ('smoothness_name', 'smoothness'),
        [('0.000005', '0.000005')],
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
        'ff_1st_block',
        (True,)
    )

    param_iterator_obj.add_pair(
        'ff_1st_bn_before_act',
        (True, False)
    )


def main_models_ns2250_generator(with_source):
    # this only contains models
    # presented in the thesis paper.

    def model_r():
        """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""
        param_iterator_obj = utils.ParamIterator()

        add_common_part_ns2250(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32,)
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2, 3)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(1, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # cm-last
                # this is different from (`cummean`, -1).
                # for loss calculation.
                # for (`cummean`, -1),
                # loss used all iterations during training, due to broadcasting.
                # but early stopping only used the last.
                #
                # by definition, we should NOT use all iterations,
                # but only the last.

                ('cummean_last', -1),
                # cm-avg
                ('cummean', 'none'),
                # inst-last
                ('last', -1),
                # inst-avg
                ('instant', 'none'),
            ],
        )

        return param_iterator_obj

    def model_additional_ff():
        param_iterator_obj = utils.ParamIterator()
        add_common_part_ns2250(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32,)
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (4, 5, 6)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(1, 2),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                ('cummean', -1),
            ],
        )

        return param_iterator_obj

    for x in chain(
            model_r().generate(),
            model_additional_ff().generate(),
    ):
        source = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
            (-1, 'cummean'): 'deep-ff',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'tang'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        x['additional_key'] = '0,500'

        assert len(x) == 27
        if with_source:
            yield source, x
        else:
            yield x


def main_models_ns2250_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in main_models_ns2250_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)
    assert len(key_all) == (
        # 480 + # 3,5 layer ff models to compare with R
        # 480 2, 3 layer FF models are actually computed
        # by avergaing multiple R=1 cases.
            288 * 24 +  # corresponding recurrent models
            # redundant r models that are the same as 3-layer FF models.
            # their performance are averaged with ff models
            144 * 4 +
            # 4,5,6 layer FF models.
            # this might be used in appendix.
            432 +
            # redundant r models that are the same as 2-layer FF models.
            # their performance are averaged with ff models
            144 * 4
    )

    # check that scripts specified in the README can indeed cover all
    # the cases.
    key_all_2nd = set()
    key_all_2nd_full = set()
    for y in chain(
            explored_models_20201002_tang_generator(),
            explored_models_20201018_tang_generator(),
    ):
        y_full = {
            'dataset_prefix': 'tang',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        y_full.update(y)

        key_y = keygen(
            # skip these two because they are of float
            **{k: v for k, v in y_full.items() if k not in {'scale', 'smoothness'}}
        )

        assert key_y not in key_all_2nd_full
        key_all_2nd_full.add(key_y)

        # remove some extra models.
        if y_full['rcnn_bl_cls'] > 7:
            continue
        if y_full['out_channel'] not in {8, 16, 32}:
            continue
        if y_full['additional_key'] != '0,500':
            continue

        assert key_y not in key_all_2nd
        key_all_2nd.add(key_y)

    assert key_all_2nd == key_all
    assert key_all_2nd_full >= key_all_2nd


def multipath_models_ns2250_generator(with_source):
    # 2L, 16/32 ch models,
    # cls 2 through 7
    def model_r():
        param_iterator_obj = utils.ParamIterator()

        add_common_part_ns2250(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32,)
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2, 3,)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(2, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # cm-last
                # this is different from (`cummean`, -1).
                # for loss calculation.
                # for (`cummean`, -1),
                # loss used all iterations during training, due to broadcasting.
                # but early stopping only used the last.
                #
                # by definition, we should NOT use all iterations,
                # but only the last.

                ('cummean_last', -1),
                # cm-avg
                ('cummean', 'none'),
                # inst-last
                ('last', -1),
                # inst-avg
                ('instant', 'none'),
            ],
        )

        return param_iterator_obj

    for x in model_r().generate():
        source = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'tang'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        x['multi_path'] = True
        x['multi_path_separate_bn'] = True
        x['additional_key'] = '0,500'

        assert len(x) == 29
        if with_source:
            yield source, x
        else:
            yield x


def multipath_models_ns2250_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in multipath_models_ns2250_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)

    # 16 variants per size.
    # 4 readout
    # 6 cls
    # 3 ch
    # 2 layer
    # 3 training size
    assert len(key_all) == 16 * 4 * 6 * 3 * 2 * 3

    # check that scripts specified in the README can indeed cover all
    # the cases.
    key_all_2nd = set()
    key_all_2nd_full = set()
    for y in chain(
            explored_models_20201215_tang_generator(),
    ):
        y_full = {
            'dataset_prefix': 'tang',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        y_full.update(y)

        key_y = keygen(
            # skip these two because they are of float
            **{k: v for k, v in y_full.items() if k not in {'scale', 'smoothness'}}
        )
        assert key_y not in key_all_2nd_full
        key_all_2nd_full.add(key_y)

        # remove some extra models.
        if y_full['rcnn_bl_cls'] > 7:
            continue
        if y_full['out_channel'] not in {8, 16, 32}:
            continue
        if not y_full['multi_path_separate_bn']:
            continue

        assert key_y not in key_all_2nd
        key_all_2nd.add(key_y)

    assert key_all_2nd == key_all
    assert key_all_2nd_full >= key_all_2nd


def ablation_models_8k_generator(with_source):
    # 2L, 16/32 ch models,
    # cls 2 through 7
    def model_r():
        """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""
        param_iterator_obj = utils.ParamIterator()

        add_common_part_8k(param_iterator_obj)

        param_iterator_obj.add_pair(
            'train_keep',
            (None,),
            replace=True
        )

        param_iterator_obj.add_pair(
            'out_channel',
            (16, 32,),
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2,)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(2, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # cm-last
                # this is different from (`cummean`, -1).
                # for loss calculation.
                # for (`cummean`, -1),
                # loss used all iterations during training, due to broadcasting.
                # but early stopping only used the last.
                #
                # by definition, we should NOT use all iterations,
                # but only the last.

                ('cummean_last', -1),
                # cm-avg
                ('cummean', 'none'),
                # inst-last
                ('last', -1),
                # inst-avg
                ('instant', 'none'),
            ],
        )

        return param_iterator_obj

    for x in model_r().generate():
        source = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'yuanyuan_8k_a_3day'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        x['multi_path'] = True
        x['multi_path_separate_bn'] = True

        for prefix in ['onlyD', 'geD', 'leD']:
            for depth_this in range(1, x['rcnn_bl_cls'] + 1):
                param_dict_ret = deepcopy(x)
                param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                assert len(param_dict_ret) == 29
                if not with_source:
                    yield param_dict_ret
                else:
                    yield source, param_dict_ret


def ablation_models_8k_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in ablation_models_8k_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)

    # 16 variants per size.
    # 4 readout
    # (2+3+4+5+6+7) per ablation
    # 2 ch
    # 1 layer
    # 1 training size
    # 3 ablation (onlyD, geD, leD)
    assert len(key_all) == 16 * 4 * (2 + 3 + 4 + 5 + 6 + 7) * 2 * 1 * 1 * 3

    # check that scripts specified in the README can indeed cover all
    # the cases.
    key_all_2nd = set()
    key_all_2nd_full = set()
    for y in chain(
            explored_models_20201205_generator(),
            explored_models_20201205_2_generator(),
            explored_models_20201213_generator(),
            explored_models_20201213_2_generator(),
            explored_models_20201221_generator(),
    ):
        y_full = {
            'dataset_prefix': 'yuanyuan_8k_a_3day',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        y_full.update(y)

        key_y = keygen(
            # skip these two because they are of float
            **{k: v for k, v in y_full.items() if k not in {'scale', 'smoothness'}}
        )

        assert key_y not in key_all_2nd_full
        key_all_2nd_full.add(key_y)

        # remove some extra models.
        if y_full['rcnn_bl_cls'] > 7:
            continue
        if y_full['out_channel'] not in {16, 32}:
            continue
        if not y_full['multi_path_separate_bn']:
            continue

        assert key_y not in key_all_2nd
        key_all_2nd.add(key_y)

    assert key_all_2nd == key_all
    assert key_all_2nd_full >= key_all_2nd


def ablation_models_ns2250_generator(with_source):
    # 2L, 16/32 ch models,
    # cls 2 through 7
    def model_r():
        """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""
        param_iterator_obj = utils.ParamIterator()

        add_common_part_ns2250(param_iterator_obj)

        param_iterator_obj.add_pair(
            'train_keep',
            (1400,),
            replace=True
        )

        param_iterator_obj.add_pair(
            'out_channel',
            (16, 32,),
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2,)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(2, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # cm-last
                # this is different from (`cummean`, -1).
                # for loss calculation.
                # for (`cummean`, -1),
                # loss used all iterations during training, due to broadcasting.
                # but early stopping only used the last.
                #
                # by definition, we should NOT use all iterations,
                # but only the last.

                ('cummean_last', -1),
                # cm-avg
                ('cummean', 'none'),
                # inst-last
                ('last', -1),
                # inst-avg
                ('instant', 'none'),
            ],
        )

        return param_iterator_obj

    for x in model_r().generate():
        source = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'tang'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        x['multi_path'] = True
        x['multi_path_separate_bn'] = True
        x['additional_key'] = '0,500'

        for prefix in ['onlyD', 'geD', 'leD']:
            for depth_this in range(1, x['rcnn_bl_cls'] + 1):
                param_dict_ret = deepcopy(x)
                param_dict_ret['multi_path_hack'] = prefix + str(depth_this)
                assert len(param_dict_ret) == 30
                if not with_source:
                    yield param_dict_ret
                else:
                    yield source, param_dict_ret


def ablation_models_ns2250_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in ablation_models_ns2250_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)

    # 16 variants per size.
    # 4 readout
    # (2+3+4+5+6+7) per ablation
    # 2 ch
    # 1 layer
    # 1 training size
    # 3 ablation (onlyD, geD, leD)
    assert len(key_all) == 16 * 4 * (2 + 3 + 4 + 5 + 6 + 7) * 2 * 1 * 1 * 3

    # check that scripts specified in the README can indeed cover all
    # the cases.
    key_all_2nd = set()
    key_all_2nd_full = set()
    for y in chain(
            explored_models_20201218_tang_generator(),
            explored_models_20201221_tang_generator(),
    ):
        y_full = {
            'dataset_prefix': 'tang',
            'model_prefix': 'maskcnn_polished_with_rcnn_k_bl',
            'yhat_reduce_pick': -1,
        }
        y_full.update(y)

        key_y = keygen(
            # skip these two because they are of float
            **{k: v for k, v in y_full.items() if k not in {'scale', 'smoothness'}}
        )

        assert key_y not in key_all_2nd_full
        key_all_2nd_full.add(key_y)

        # remove some extra models.
        if y_full['rcnn_bl_cls'] > 7:
            continue
        if y_full['out_channel'] not in {16, 32}:
            continue
        if not y_full['multi_path_separate_bn']:
            continue

        assert key_y not in key_all_2nd
        key_all_2nd.add(key_y)

    assert key_all_2nd == key_all
    assert key_all_2nd_full >= key_all_2nd


def ablation_ff_models_8k_generator(with_source):
    # inst-last, only last iteration kept. basically deep models.
    def model_r():
        """those in scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/submit_20200430.py"""
        param_iterator_obj = utils.ParamIterator()

        add_common_part_8k(param_iterator_obj)

        param_iterator_obj.add_pair(
            'out_channel',
            (8, 16, 32, 48, 64),
        )

        param_iterator_obj.add_pair(
            'num_layer',
            (2, 3,)
        )

        param_iterator_obj.add_pair(
            'rcnn_bl_cls',
            range(2, 8),
        )

        param_iterator_obj.add_pair(
            ('rcnn_acc_type', 'yhat_reduce_pick',),
            [
                # inst-last
                ('last', -1),
            ],
        )

        return param_iterator_obj

    for x in model_r().generate():
        source = {
            (-1, 'last'): 'inst-last',
        }[x['yhat_reduce_pick'], x['rcnn_acc_type']]

        x['dataset_prefix'] = 'yuanyuan_8k_a_3day'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'
        x['multi_path'] = True
        x['multi_path_separate_bn'] = True

        x['multi_path_hack'] = 'onlyD' + str(x['rcnn_bl_cls'])
        assert len(x) == 29
        if not with_source:
            yield x
        else:
            yield source, x


def ablation_ff_models_8k_validate():
    # check that the list of scripts
    # in the README covers all main models.
    key_all = set()
    for x in ablation_ff_models_8k_generator(with_source=False):
        key = keygen(
            # skip these two because they are of float
            **{k: v for k, v in x.items() if k not in {'scale', 'smoothness'}}
        )
        assert key not in key_all
        key_all.add(key)

    # 16 variants per size.
    # 1 readout
    # 6 cls
    # 5 ch
    # 2 layer
    # 3 training size
    assert len(key_all) == 16 * 1 * 6 * 5 * 2 * 3
