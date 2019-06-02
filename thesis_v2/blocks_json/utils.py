from collections import OrderedDict


def update_module_dict(old_dict, new_dict):
    assert type(old_dict) is OrderedDict
    assert type(new_dict) is OrderedDict
    assert old_dict.keys() & new_dict.keys() == set()
    old_dict.update(old_dict)


def generate_param_dict(*,
                        module_dict,
                        op_params=None,
                        ):
    if op_params is None:
        # simplest one
        op_spec_list = [
            {'name': 'module',
             'args': [x],
             'kwargs': {},
             } for x in module_dict.keys()
        ]

        op_list = [
            {
                'name': 'sequential',
                'args': [op_spec_list, ],
                'kwargs': {},
                'in': 'input0',
                'out': 'out_neural',
            },
        ]

        param_dict = {
            'module_dict': module_dict,
            'op_list': op_list,
            'out': ['out_neural', ],
        }
    else:
        raise NotImplementedError

    return param_dict


def new_map_size(map_size, kernel_size, padding, stride):
    # TODO: dilation support.
    map_size_new = ((map_size[0] - kernel_size + 2 * padding) // stride + 1,
                    (map_size[1] - kernel_size + 2 * padding) // stride + 1)
    assert (map_size_new[0] - 1) * stride + kernel_size == map_size[
        0] + 2 * padding
    assert (map_size_new[1] - 1) * stride + kernel_size == map_size[
        1] + 2 * padding
    # print(map_size_new)
    return map_size_new


def check_input_size(input_size):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    else:
        input_size = tuple(input_size)

    assert isinstance(input_size, tuple) and len(input_size) == 2
    assert type(input_size[0]) is int and input_size[0] > 0
    assert type(input_size[1]) is int and input_size[1] > 0
    return input_size
