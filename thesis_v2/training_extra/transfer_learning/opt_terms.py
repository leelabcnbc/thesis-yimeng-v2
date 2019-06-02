from ..typecheck import type_check_wrapper


def sanity_check_opt_config(config):
    return type_check_wrapper(config, _type_checker,
                              {'fc', 'loss', 'optimizer'})


def sanity_check_one_fc_layer_opt_config(config):
    return type_check_wrapper(config, _type_checker, {'sparse'})


def sanity_check_one_optimizer_opt_config(config):
    assert isinstance(config, dict)
    optimizer_type = config['optimizer_type']
    if optimizer_type == 'sgd':
        assert type_check_wrapper(config, _type_checker, {'optimizer_type',
                                                          'lr', 'momentum'})
    elif optimizer_type == 'adam':
        assert type_check_wrapper(config, _type_checker, {'optimizer_type',
                                                          'lr'})
    else:
        raise NotImplementedError
    return True


def generate_one_optimizer_config(optimizer_type, lr=None):
    config = {'optimizer_type': optimizer_type}
    if optimizer_type == 'sgd':
        config['lr'] = 0.01 if lr is None else lr
        config['momentum'] = 0.9
    elif optimizer_type == 'adam':
        config['lr'] = 0.001 if lr is None else lr
    else:
        raise NotImplementedError

    assert sanity_check_one_optimizer_opt_config(config)
    return config


def generate_one_opt_config(fc_config, loss, optimizer):
    config = dict()
    config['fc'] = fc_config
    config['loss'] = loss
    config['optimizer'] = optimizer
    assert sanity_check_opt_config(config)
    return config


def generate_one_fc_layer_opt_config(sparse):
    # by default, l1_bias = l1, l2_bias = l2.
    # so to stop them from learning you need to set them to zero.

    config = dict()
    config['sparse'] = sparse

    assert sanity_check_one_fc_layer_opt_config(config)
    return config


_type_checker = {
    'sparse': float,
    'fc': sanity_check_one_fc_layer_opt_config,
    'loss': lambda x: x in {'mse', 'poisson'},
    'optimizer': sanity_check_one_optimizer_opt_config,
    'optimizer_type': lambda x: x in {'adam', 'sgd'},
    'lr': float,
    'momentum': float,
}
