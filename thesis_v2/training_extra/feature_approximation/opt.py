from torch import nn, optim

from .opt_terms import (
    generate_one_opt_config,
    generate_one_optimizer_config,
    sanity_check_one_optimizer_opt_config
)


# noinspection PyUnusedLocal
def get_feature_approximation_opt_config(*, model_json: dict = None,
                                         loss_type='mse'):
    # later on, model_json might be needed to get shapes of weight, etc.

    return generate_one_opt_config(
        loss_type,
        generate_one_optimizer_config('adam'),
    )


def get_optimizer(model: nn.Module, optimizer_config: dict):
    assert sanity_check_one_optimizer_opt_config(optimizer_config)
    params_to_learn = model.parameters()

    if optimizer_config['optimizer_type'] == 'sgd':
        optimizer_this = optim.SGD(params_to_learn,
                                   lr=optimizer_config['lr'],
                                   momentum=optimizer_config['momentum'])
    elif optimizer_config['optimizer_type'] == 'adam':
        optimizer_this = optim.Adam(params_to_learn,
                                    lr=optimizer_config['lr'])
    else:
        raise NotImplementedError
    return optimizer_this
