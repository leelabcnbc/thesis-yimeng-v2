from collections import OrderedDict


def update_module_dict(old_dict: OrderedDict, new_dict: OrderedDict) -> None:
    assert type(old_dict) is OrderedDict
    assert type(new_dict) is OrderedDict
    assert old_dict.keys() & new_dict.keys() == set()
    old_dict.update(new_dict)


def generate_param_dict(*,
                        module_dict,
                        op_params=None,
                        comments=None,
                        output_list=True,
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
        # a bunch of sequential stuffs.
        assert type(op_params) is list
        op_list = []
        for op_param_dict in op_params:
            assert op_param_dict.keys() == {'type', 'param', 'in', 'out', 'keep_out'}
            if op_param_dict['type'] == 'sequential':
                # this is only one supported
                # op_param_dict['param'] is a function
                if callable(op_param_dict['param']):
                    predicate = op_param_dict['param']
                    seq_op_args = dict()
                else:
                    # it should be an iterable with 2 elements
                    predicate, seq_op_args = op_param_dict['param']
                assert seq_op_args.keys() <= {'module_op_name'}
                module_op_name = seq_op_args.get('module_op_name', 'module')
                op_spec_list = [{
                    'name': module_op_name,
                    'args': [x],
                    'kwargs': {},
                } for idx, x in enumerate(module_dict.keys()) if op_param_dict['param'](idx, x)
                ]
                op_list.append({
                    'name': 'sequential',
                    'args': [op_spec_list, ],
                    'kwargs': {},
                    'in': op_param_dict['in'],
                    'out': op_param_dict['out'],
                })
            elif op_param_dict['type'] == 'stack':
                assert op_param_dict['param'].keys() <= {'dim'}
                op_list.append({
                    'name': 'stack',
                    'args': [],
                    'kwargs': {'dim': op_param_dict['param'].get('dim', 0)},
                    'in': op_param_dict['in'],
                    'out': op_param_dict['out'],
                })
            else:
                raise NotImplementedError

        param_dict = {
            'module_dict': module_dict,
            'op_list': op_list,
            'out': [z['out'] for z in op_params if z['keep_out']],
        }

    if not output_list:
        assert len(param_dict['out']) == 1
        param_dict['out'] = param_dict['out'][0]

    if comments is not None:
        param_dict['comments'] = comments

    return param_dict


def new_map_size(map_size, kernel_size, padding, stride, strict=True,
                 ceil_mode=False):
    map_size_new = ((map_size[0] - kernel_size + 2 * padding +
                     (0 if not ceil_mode else stride - 1)) // stride + 1,
                    (map_size[1] - kernel_size + 2 * padding +
                     (0 if not ceil_mode else stride - 1)) // stride + 1)
    if strict:
        assert not ceil_mode
        assert (map_size_new[0] - 1) * stride + kernel_size == map_size[
            0] + 2 * padding
        assert (map_size_new[1] - 1) * stride + kernel_size == map_size[
            1] + 2 * padding
    # print(map_size_new)
    return map_size_new


def check_input_size(input_size, strict=False, sequence_type='tuple'):
    if sequence_type == 'tuple':
        sequence_type_obj = tuple
    elif sequence_type == 'list':
        sequence_type_obj = list
    else:
        raise NotImplementedError

    if not strict:
        if isinstance(input_size, int):
            input_size = sequence_type_obj((input_size, input_size))
        else:
            input_size = sequence_type_obj(input_size)
    assert type(input_size) is sequence_type_obj and len(input_size) == 2
    assert type(input_size[0]) is int and input_size[0] > 0
    assert type(input_size[1]) is int and input_size[1] > 0
    return input_size
