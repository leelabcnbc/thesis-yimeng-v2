from collections import OrderedDict
from typing import Tuple, Optional
from .utils import check_input_size, update_module_dict, new_map_size
from . import general


def check_conv2dstack_config(
        *,
        bn: bool,
        bn_affine: Optional[bool],
        bn_before_act: Optional[bool],
        act_fn: Optional[str],
):
    if not bn:
        # bn_before_act makes no sense.
        assert bn_before_act is None
        assert bn_affine is None
    elif act_fn is None:
        # bn = True, NO act.
        assert bn_before_act is None
        assert bn_affine is not None
    elif act_fn is not None:
        # bn=True, has act
        assert bn_before_act is not None
        assert bn_affine is not None
    else:
        raise RuntimeError


def conv2dstack(*,
                input_size: Tuple[int, int],
                suffix: str,
                kernel_size: int,
                in_channel: int,
                out_channel: int, bn: bool = True,
                bn_affine: Optional[bool] = True,
                act_fn: Optional[str],
                init_std: float = 0.01,
                do_init: bool = True,
                bn_before_act: Optional[bool] = True,
                padding: int = 0,
                # this is needed to generate the new map size
                state_dict: dict,
                ):
    """
    a standard conv stack with conv + bn + act_fn (bn/act_fn can switch order).
    """
    assert type(input_size) is tuple
    assert type(suffix) is str
    assert type(kernel_size) is int
    assert type(in_channel) is int
    assert type(out_channel) is int
    assert type(bn) is bool
    assert (type(bn_affine) is bool) or (bn_affine is None)
    assert (type(act_fn) is str) or (act_fn is None)
    assert type(init_std) is float
    assert type(do_init) is bool
    assert (type(bn_before_act) is bool) or (bn_before_act is None)
    assert type(padding) is int
    assert type(state_dict) is dict

    check_conv2dstack_config(bn=bn, bn_affine=bn_affine, bn_before_act=bn_before_act, act_fn=act_fn)
    input_size = check_input_size(input_size, strict=True)
    module_dict = OrderedDict()
    module_dict['conv' + suffix] = {
        'name': 'torch.nn.conv2d',
        'params': {'in_channels': in_channel, 'out_channels': out_channel,
                   'kernel_size': kernel_size, 'padding': padding,
                   # if we don't use bn, or we have bn after act, then we should have the bias.
                   'bias': (not bn) or (bn and (bn_before_act is False))
                   },
        'init': {
            'strategy': 'normal',
            'parameters': {'std': init_std}
        } if do_init else None,
    }

    def add_bn():
        update_module_dict(module_dict,
                           general.bn(name='bn' + suffix,
                                      num_features=out_channel,
                                      affine=bn_affine, do_init=do_init)
                           )

    def add_act():
        update_module_dict(module_dict,
                           general.act(name='act' + suffix,
                                       act_fn=act_fn)
                           )

    if bn and (act_fn is not None):
        if bn_before_act:
            add_bn()
            add_act()
        else:
            add_act()
            add_bn()
    elif bn and (act_fn is None):
        # only one is available.
        add_bn()
    elif (not bn) and (act_fn is not None):
        add_act()
    elif (not bn) and (act_fn is None):
        pass
    else:
        raise RuntimeError

    state_dict['map_size'] = new_map_size(input_size, kernel_size, padding, 1)

    return module_dict
