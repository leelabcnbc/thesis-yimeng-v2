from collections import OrderedDict
from typing import Tuple, Optional
from .utils import check_input_size


def factoredfc(*,
               name: str,
               map_size: Tuple[int, int],
               in_channels: int,
               out_features: int,
               init_std: float = 0.01,
               do_init=True,
               weight_feature_constraint: Optional[str] = None,
               weight_spatial_constraint: Optional[str] = None,
               bias: bool = True,
               ):
    assert type(name) is str
    assert type(map_size) is tuple
    assert type(in_channels) is int
    assert type(out_features) is int
    assert type(init_std) is float
    assert type(do_init) is bool
    assert (type(weight_feature_constraint) is str) or (weight_feature_constraint is None)
    assert (type(weight_spatial_constraint) is str) or (weight_spatial_constraint is None)

    module_dict = OrderedDict()
    check_input_size(map_size, strict=True)
    module_dict[name] = {
        'name': 'maskcnn.factoredfc',
        'params': {
            'in_channels': in_channels,
            'map_size': [map_size[0], map_size[1]],
            'out_features': out_features,
            'bias': bias,
            'weight_feature_constraint': weight_feature_constraint,
            'weight_spatial_constraint': weight_spatial_constraint,
        },
        'init': {
            'strategy': 'normal',
            'parameters': {'std': init_std}
        } if do_init else None
    }

    return module_dict
