from collections import OrderedDict
from typing import Tuple


def factoredfc(*,
               name: str,
               map_size: Tuple[int, int],
               in_channels: int,
               out_features: int,
               init_std: float = 0.01,
               do_init=True,
               ):
    module_dict = OrderedDict()
    assert type(map_size) is tuple and len(map_size) == 2
    for x in map_size:
        assert type(x) is int and x > 0
    module_dict[name] = {
        'name': 'maskcnn.factoredfc',
        'params': {
            'in_channels': in_channels,
            'map_size': [map_size[0], map_size[1]],
            'out_features': out_features,
            'bias': True,
        },
        'init': {
            'strategy': 'normal',
            'parameters': {'std': init_std}
        } if do_init else None
    }

    return module_dict
