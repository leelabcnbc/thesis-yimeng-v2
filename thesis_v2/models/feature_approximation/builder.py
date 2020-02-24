"""generates the config file needed for approximating feature"""

from typing import List
from collections import OrderedDict
from ...blocks_json import general, utils, conv
from ...blocks_json.utils import update_module_dict, generate_param_dict


def gen_local_pcn_recurrent_feature_approximator(
        *,
        in_shape_lower: List[int],
        in_shape_higher: List[int],
        kernel_size=5,  # 5 for one iteration, 9 for two iterations, 13 for three iterations.
        act_fn='relu',
        batchnorm_pre=True,
        batchnorm_post=True,
        batchnorm_affine=True,
        do_init=True,
):
    # in_shape_lower is (C1, H, W).
    # in_shape_higher is (C2, H, W).
    # this simplifies the design of optimizer.

    assert len(in_shape_lower) == len(in_shape_higher) == 3
    assert in_shape_lower[1:] == in_shape_higher[1:]

    module_dict = OrderedDict()

    if batchnorm_pre:
        # add pre batch norm
        update_module_dict(module_dict, general.bn(
            name='bn',
            num_features=in_shape_lower[0] + in_shape_higher[0],
            affine=batchnorm_affine,
            do_init=do_init,
        ))

    # then a conv stack
    assert kernel_size % 2 == 1
    state_dict = dict()
    utils.update_module_dict(module_dict,
                             conv.conv2dstack(
                                 input_size=(in_shape_lower[1], in_shape_lower[2]),
                                 suffix='0',
                                 kernel_size=kernel_size,
                                 padding=kernel_size // 2,
                                 in_channel=in_shape_lower[0] + in_shape_higher[0],
                                 out_channel=in_shape_higher[0],
                                 act_fn=act_fn,
                                 # `bn_before_act` must be False, so that BN is afterwards, allowing negative values
                                 # to be modeled.
                                 bn_before_act=False,
                                 state_dict=state_dict,
                                 bn=batchnorm_post,
                                 do_init=do_init,
                             ))
    assert state_dict['map_size'] == (in_shape_higher[1], in_shape_higher[2])

    return generate_param_dict(module_dict=module_dict,
                               op_params=None)
