from collections import OrderedDict
from typing import Tuple, Optional
from .utils import check_input_size, new_map_size


def pool2d(
        *,
        name: str,
        pooling_type: str,
        kernel_size: int,
        stride: Optional[int] = None,
        input_size: Tuple[int, int],
        state_dict: dict,
        ceil_mode: bool = False,
        map_size_strict: bool = True,
) -> OrderedDict:
    assert type(name) is str
    assert type(pooling_type) is str
    assert type(kernel_size) is int
    assert (type(stride) is int) or (stride is None)
    assert type(input_size) is tuple
    assert type(state_dict) is dict
    assert type(ceil_mode) is bool
    assert type(map_size_strict) is bool

    check_input_size(input_size, strict=True)

    module_dict = OrderedDict()

    module_dict[name] = {
        'name': {'max': 'torch.nn.maxpool2d',
                 'avg': 'torch.nn.avgpool2d'}[pooling_type],
        'params': {'kernel_size': kernel_size,
                   # preserve everything.
                   # I checked that actually for those border,
                   # no extra -inf or 0 will be added.
                   # say you do kernel-size-3 1D mean pooling
                   # on a tensor of shape 8, then output size is 3,
                   # with last element of output
                   # computed as mean of last two elements
                   # in the input (not having a padded 0 and then divide
                   # by 3).
                   'ceil_mode': ceil_mode},
        'init': None,
    }

    state_dict['map_size'] = new_map_size(input_size, kernel_size, 0,
                                          kernel_size,
                                          strict=map_size_strict, ceil_mode=ceil_mode)

    return module_dict
