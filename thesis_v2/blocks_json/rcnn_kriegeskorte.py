from collections import OrderedDict
from typing import Tuple, List, Optional
from .utils import check_input_size, new_map_size


def blstack(
        *,
        name: str,
        input_size: Tuple[int, int],
        n_timesteps: int,
        channel_list: List[int],
        kernel_size_list: List[int],
        pool_ksize: int,
        pool_type: Optional[str],
        act_fn: str,
        init_std: float = 0.01,
        do_init: bool = True,
        state_dict: dict,
        norm_type: str,
        # multi-path ensemble
        multi_path: bool = False,
        # if `multi_path_separate_bn` is true, each path has its own BNs;
        # otherwise, they share some BNs.
        multi_path_separate_bn: Optional[bool] = None,
        multi_path_hack: Optional[str] = None,
):
    input_size = check_input_size(input_size, strict=True)

    assert type(kernel_size_list) is list
    assert type(channel_list) is list
    assert len(kernel_size_list) + 1 == len(channel_list)

    map_size = input_size

    for layer_idx, k in enumerate(kernel_size_list):
        assert type(k) is int
        assert k > 0 and k % 2 == 1
        # pooling for layer_idx > 0
        if layer_idx != 0:
            map_size = new_map_size(map_size, pool_ksize, 0, pool_ksize, strict=False, ceil_mode=True)
        # conv does not change result
        assert new_map_size(map_size, k, k // 2, 1) == map_size

    for c in channel_list:
        assert type(c) is int
        assert c > 0

    assert type(init_std) is float
    assert type(pool_ksize) is int and pool_ksize > 0

    module_dict = OrderedDict()

    module_dict[name] = {
        'name': 'rcnn_kriegeskorte.blstack',
        'params': {
            'n_timesteps': n_timesteps,
            'channel_list': channel_list,
            'ksize_list': kernel_size_list,
            'pool_ksize': pool_ksize,
            'pool_type': pool_type,
            'act_fn': act_fn,
            'norm_type': norm_type,
            'multi_path': multi_path,
            'multi_path_separate_bn': multi_path_separate_bn,
            'multi_path_hack': multi_path_hack,
        },
        'init': {
            'strategy': 'normal',
            'parameters': {'std': init_std}
        } if do_init else None,
    }

    # update map size.
    state_dict['map_size'] = map_size

    return module_dict


def accumulator(
        *,
        name: str,
        mode: str,
):
    module_dict = OrderedDict()
    module_dict[name] = {
        'name': 'rcnn_kriegeskorte.accumulator',
        'params': {'mode': mode},
        'init': None,
    }

    return module_dict
