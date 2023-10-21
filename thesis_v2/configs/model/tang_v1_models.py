from itertools import chain
from copy import deepcopy

from ...submission import utils

readout_type_mapping = {
    'inst-last': 'no-avg',
    'cm-last': 'early-avg',
    'inst-avg': 'late-avg',
    'cm-avg': '2-avg',
}

def add_common_part_tang_v1(param_iterator_obj):
    param_iterator_obj.add_pair(
        'train_keep',
        # (1280, 2560, None),
        (None, ),
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
        'px_kept',
        (100,)
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
    
def explored_models_20220923(with_source):

    def model_r():
        param_iterator_obj = utils.ParamIterator()

        add_common_part_tang_v1(param_iterator_obj)

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
        add_common_part_tang_v1(param_iterator_obj)

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

        x['dataset_prefix'] = 'tang_m3s1'
        x['model_prefix'] = 'maskcnn_polished_with_rcnn_k_bl'

        assert len(x) == 27
        if with_source:
            yield source, x
        else:
            yield x

def explored_models_20230615(with_source):
    for src, param_dict in explored_models_20220923(True):
        param_dict_ret = deepcopy(param_dict)
        param_dict_ret['train_keep'] = 17450
        if not with_source:
            yield param_dict_ret
        else:
            yield src, param_dict_ret
            
def explored_models_20230626(with_source):
    for src, param_dict in explored_models_20220923(True):
        param_dict_ret = deepcopy(param_dict)
        param_dict_ret['train_keep'] = 8725
        if not with_source:
            yield param_dict_ret
        else:
            yield src, param_dict_ret