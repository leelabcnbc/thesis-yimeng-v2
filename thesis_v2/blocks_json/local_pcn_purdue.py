from collections import OrderedDict
from typing import Tuple, Optional
from .utils import check_input_size, update_module_dict, new_map_size
from . import general


def conv2d_baseline(
        *,
        input_size: Tuple[int, int],
        suffix: str,
        kernel_size: int,
        in_channel: int,
        out_channel: int,
        bn: bool,
        bn_affine: Optional[bool] = True,
        act_fn: Optional[str],
        init_std: float = 0.01,
        do_init: bool = True,
        bn_post: bool,
        final_act: bool,
        padding: int,
        # this is needed to generate the new map size,

        localpcn_bypass: bool,

        # this is not implemented; just left here so later can be explored.
        localpcn_tied: bool,
        localpcn_b0_init: float,
        localpcn_no_act: bool,
        localpcn_cls: int,
        localpcn_bias: bool,

        state_dict: dict,

        bn_locations_legacy: bool,
        # if True,
        # bn -> conv -> bn_post -> act
        # to match what's used before.

        # if False,
        # it's conv -> bn -> act -> bn_post.
        # this may interface better with maskcnn_polished.

):
    assert kernel_size % 2 == 1
    assert padding == kernel_size // 2  # this makes things easy.

    input_size = check_input_size(input_size, strict=True)

    module_dict = OrderedDict()

    def add_bn(prefix, num_feature):
        update_module_dict(module_dict,
                           general.bn(name=prefix + suffix,
                                      num_features=num_feature,
                                      affine=bn_affine, do_init=do_init)
                           )

    if bn and bn_locations_legacy:
        add_bn('bn', in_channel)

    if not localpcn_tied:
        module_dict['conv' + suffix] = {
            'name': 'localpcn_purdue.baseline',
            'params': {
                'inchan': in_channel,
                'outchan': out_channel,
                'kernel_size': kernel_size,
                'padding': padding,
                'bias': localpcn_bias,
                'cls': localpcn_cls,
                'act_fn': None if localpcn_no_act else act_fn,
                'bypass': localpcn_bypass,
                'no_act': localpcn_no_act,  # when this is True, act_fn only works when there is final_act.
                'b0_init': localpcn_b0_init,
            },
            'init': {
                'strategy': 'normal',
                'parameters': {'std': init_std}
            } if do_init else None,
        }
    else:
        raise NotImplementedError

    if bn_post and bn_locations_legacy:
        add_bn('bn_post', out_channel)

    if bn and (not bn_locations_legacy):
        add_bn('bn', out_channel)

    if final_act and act_fn is not None:
        update_module_dict(module_dict,
                           general.act(name='act' + suffix,
                                       act_fn=act_fn)
                           )

    if bn_post and (not bn_locations_legacy):
        add_bn('bn_post', out_channel)

    # update map size.
    state_dict['map_size'] = new_map_size(input_size, kernel_size, padding, 1)
    assert state_dict['map_size'] == input_size

    return module_dict
