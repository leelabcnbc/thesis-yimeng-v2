from collections import OrderedDict


def bn(*,
       name: str,
       num_features: int,
       affine: bool,
       eps: float = 0.001,
       momentum: float = 0.1,
       do_init: bool = True,
       ):
    module_dict = OrderedDict()
    module_dict[name] = {
        'name': 'torch.nn.batchnorm2d',
        # simply normalize everything.
        'params': {'num_features': num_features,
                   'eps': eps, 'momentum': momentum,
                   'affine': affine},
        'init': {} if do_init else None,
    }

    return module_dict


def act(*,
        name,
        act_fn,
        ):
    assert act_fn in {'softplus', 'relu', 'exp'}
    module_dict = OrderedDict()
    module_dict[name] = {
        'name': f'torch.nn.{act_fn}' if act_fn != 'exp' else 'general.exp',
        'params': {},
        'init': None
    }

    return module_dict

