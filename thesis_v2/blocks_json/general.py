from collections import OrderedDict


def bn(*,
       name: str,
       num_features: int,
       affine: bool,
       eps: float = 0.001,
       momentum: float = 0.1,
       do_init: bool = True,
       ndim: int = 2,
       ):
    # should be better named bn2d
    assert type(name) is str
    assert type(num_features) is int
    assert type(affine) is bool
    assert type(eps) is float
    assert type(momentum) is float
    assert type(do_init) is bool
    assert type(ndim) is int

    module_dict = OrderedDict()
    module_dict[name] = {
        'name': {1: 'torch.nn.batchnorm1d', 2: 'torch.nn.batchnorm2d'}[ndim],
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
    assert type(act_fn) is str

    assert act_fn in {'softplus', 'relu', 'exp', 'sigmoid', 'tanh'}
    module_dict = OrderedDict()
    module_dict[name] = {
        'name': f'torch.nn.{act_fn}' if act_fn != 'exp' else 'general.exp',
        'params': {},
        'init': None
    }

    return module_dict
