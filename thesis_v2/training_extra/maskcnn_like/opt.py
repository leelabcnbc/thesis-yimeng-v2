"""
modify original cnn_opt.py
in https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/thesis_proposal/maskcnn/cnn_opt.py  # noqa: E501
so that it can be used with JSONNet.
"""
from copy import deepcopy

from torch import optim

from torchnetjson.net import JSONNet

from .opt_terms import (
    generate_one_conv_layer_opt_config,
    generate_one_opt_config,
    generate_one_optimizer_config,
    generate_one_fc_layer_opt_config,
    sanity_check_one_optimizer_opt_config
)


def get_maskcnn_v1_opt_config(*, model_json: dict,
                              group=0.05, smoothness=0.03, scale=0.02,
                              legacy=False, loss_type='poisson'):
    layer = len(model_json['comments']['conv_layers'])

    # about `first_conv`:
    #   we want to handle those sparse coding stuff as well.
    #   in those cases (especially the precomputed one),
    #   there is no conv1 at all.

    assert layer >= 1
    conv1_config = generate_one_conv_layer_opt_config(0.0, smoothness)
    conv2_higher_config = generate_one_conv_layer_opt_config(group, 0.0)

    configs = [conv1_config, ] + [deepcopy(conv2_higher_config) for _ in
                                  range(layer - 1)]

    return generate_one_opt_config(
        configs,
        generate_one_fc_layer_opt_config(scale),
        loss_type,
        generate_one_optimizer_config('adam'),
        legacy=legacy
    )


def get_optimizer(model: JSONNet, optimizer_config: dict):
    assert sanity_check_one_optimizer_opt_config(optimizer_config)

    # I can here use model.get_param_dict()

    param_dict = model.get_param_dict()
    special_lr_config = param_dict['comments'].get(
        'special_lr_config', None
    )

    if special_lr_config is None:
        params_to_learn = model.parameters()
    else:
        raise NotImplementedError

    if optimizer_config['optimizer_type'] == 'adam':
        optimizer_this = optim.Adam(params_to_learn, lr=optimizer_config['lr'])
    else:
        raise NotImplementedError
    return optimizer_this
