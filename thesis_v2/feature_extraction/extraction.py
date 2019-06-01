from functools import partial
from typing import Callable, Tuple, Union, List
from copy import deepcopy
from json import dumps, loads

import numpy as np
import h5py
from torch import nn, Tensor, no_grad


class Lambda(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def _forward_hook(m, in_, out_: Tensor, module_name, callback_dict,
                  slice_this, verbose):
    assert isinstance(out_, Tensor)

    data_all: np.ndarray = out_.detach().cpu().numpy()
    # then slice it
    data_this_to_use: np.ndarray = data_all[(...,) + slice_this]

    if verbose:
        print(module_name, type(m))
        print('input_shape', [tuple(x.size()) for x in in_],
              'output_shape', tuple(out_.size()),
              'slice', slice_this,
              'sliced_shape', data_this_to_use.shape)

    # print(f'{data_all.shape} -> {data_this_to_use.shape}')
    # extra copy to guard against weird things,
    # also, make sure order is right.
    callback_dict[module_name].append(data_this_to_use.copy(order='C'))


def augment_module(net: nn.Module, *,
                   module_names: list,
                   name_mapping: dict,
                   slice_dict: dict, verbose=False) -> (
        dict, list):
    callback_dict = dict()
    module_names = set(module_names)

    forward_hook_remove_func_list = []
    for x, y in net.named_modules():
        preferred_name = name_mapping.get(x, x)
        if preferred_name in module_names:
            callback_dict[preferred_name] = []
            forward_hook_remove_func_list.append(
                y.register_forward_hook(
                    partial(_forward_hook,
                            module_name=preferred_name,
                            callback_dict=callback_dict,
                            # by default, take all
                            slice_this=slice_dict.get(
                                preferred_name,
                                ()
                            ),
                            verbose=verbose
                            )))

    def remove_handles():
        for h in forward_hook_remove_func_list:
            h.remove()

    return callback_dict, remove_handles


def clear_callback(callback_dict: dict):
    # do this when you fetch every batch.
    for x in callback_dict:
        callback_dict[x].clear()


def normalize_augment_config(augment_module_config: dict):
    assert augment_module_config.keys() <= {'module_names',
                                            'name_mapping',
                                            'slice_dict'}

    module_names = augment_module_config['module_names']
    name_mapping = augment_module_config.get('name_mapping', None)
    slice_dict = augment_module_config.get('slice_dict', None)

    assert type(module_names) is list
    for z in module_names:
        assert type(z) is str

    if name_mapping is None:
        # this maps internal PyTorch name to standard names (in Caffe).
        name_mapping = {}

    if slice_dict is None:
        slice_dict = {}

    assert type(name_mapping) is dict and type(slice_dict) is dict

    for k, v in name_mapping.items():
        assert type(k) is str and type(v) is str

    for k2, v2 in slice_dict.items():
        # tuple of slices
        assert type(k2) is str and type(v2) is tuple
        for v3 in v2:
            assert type(v3) is slice

    return {
        'module_names': module_names,
        'name_mapping': name_mapping,
        'slice_dict': slice_dict,
    }


def _check_int(x):
    # https://stackoverflow.com/a/37727662/3692822
    # https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not  # noqa: E501
    if x is not None:
        assert np.issubsctype(type(x), np.integer)
        return int(x)
    else:
        return x


def jsonify_augment_config(config: dict):
    # copy so we get a new one.
    config = deepcopy(config)
    slice_dict = config['slice_dict']

    # `config['slice_dict']` changed in place.
    for k in slice_dict:
        v_old = slice_dict[k]
        slice_dict[k] = [
            {
                # otherwise it can be int64
                'start': _check_int(x.start),
                'stop': _check_int(x.stop),
                'step': _check_int(x.step),
            } for x in v_old
        ]

    assert loads(dumps(config)) == config

    return config


def _extract_features_save(module_names_ordered: List[str],
                           output_group: h5py.Group,
                           start_idx: int,
                           stop_idx: int,
                           numel: int,
                           callback_dict: dict,
                           ):
    for mod_idx, mod_name in enumerate(module_names_ordered):
        data_array = callback_dict[mod_name]
        # should be a list
        for recurrent_idx, data in enumerate(data_array):
            name = f'{mod_idx}.{recurrent_idx}'
            if name not in output_group:
                output_group.create_dataset(
                    name,
                    shape=(numel,) + data.shape[1:],
                    dtype=data.dtype,
                    compression="gzip"
                )
            # then store
            output_group[name][start_idx:stop_idx] = data

    # flushing is disabled, for best performance?
    # output_group.file.flush()


def extract_features(net: nn.Module,
                     datasets: Tuple[Union[np.ndarray, h5py.Dataset], ...],
                     *,
                     preprocessor: Callable,  # turn data to be net-ready.
                     output_group: h5py.Group,  # where to write.
                     batch_size: int = 50,
                     augment_config: dict,
                     verbose=True,
                     unpack=True,
                     deterministic=True,
                     ):
    from torch.backends import cudnn
    # make sure it's reproducible on the platform.
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = deterministic

    # check shape
    num_element = datasets[0].shape[0]
    for dataset in datasets[1:]:
        assert dataset.shape[0] == num_element

    augment_config = normalize_augment_config(augment_config)
    augment_config_json = dumps(jsonify_augment_config(augment_config),
                                # for more determinism
                                sort_keys=True)

    callback_dict, remove_handles = augment_module(net,
                                                   **augment_config)

    num_batch = (num_element + (batch_size - 1)) // batch_size

    if verbose:
        print('num batch', num_batch)

    # save config, sorted key for some kind of stability.
    output_group.attrs['config'] = np.string_(augment_config_json)

    for batch_idx, start_idx in enumerate(range(0, num_element, batch_size)):
        stop_idx = min(start_idx + batch_size, num_element)

        if verbose:
            print(f'working on {start_idx} to {stop_idx} of {num_element}')

        # fetch the data. fortunately, numpy and h5py share same syntax.
        inputs = tuple(dt[slice(start_idx, stop_idx)] for dt in datasets)
        # then call preprocessor to convert it into network ready format
        # here, we can preprocess data on the fly.
        inputs = preprocessor(inputs)

        with no_grad():  # pure inference mode.
            if unpack:
                net(*inputs)
            else:
                net(inputs)

        # then save, according to module_names_ordered
        _extract_features_save(augment_config['module_names'], output_group,
                               start_idx, stop_idx, num_element,
                               callback_dict)
        # then clear
        clear_callback(callback_dict)

    # clean up
    remove_handles()
