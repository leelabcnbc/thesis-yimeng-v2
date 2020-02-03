from functools import partial

import numpy as np
import torch

from torchnetjson.net import JSONNet

"""training should invoke this function"""

from .opt import get_optimizer
from .loss import get_loss

from ..data import get_resp_train

from ..init import init_bias_wrapper, check_init_type

from ..training import train_one_wrapper


def train_one(*,
              arch_json_partial, opt_config_partial, datasets,
              key,
              show_every=1000,
              model_seed=0, train_seed=0,
              max_epoch=20000,
              early_stopping_field='loss_no_reg',
              device='cuda',
              val_test_every=50,
              return_model=True,
              extra_params=None):
    if model_seed is not None:
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
    # initialize
    # arch_json_partial is a function with two args.
    # first being input shape
    # second being number of neurons.

    # get all jsons ready (model_json, opt_config)

    return train_one_wrapper(
        get_json_fn=partial(
            get_json_fn,
            arch_json_partial=arch_json_partial,
            opt_config_partial=opt_config_partial,
        ),
        initialize_model_fn=initialize_model,
        get_optimizer_fn=get_optimizer,
        get_loss_fn=partial(get_loss, device=device),
        datasets=datasets,
        key=key,
        show_every=show_every,
        model_seed=model_seed, train_seed=train_seed,
        max_epoch=max_epoch,
        early_stopping_field=early_stopping_field,
        device=device,
        val_test_every=val_test_every,
        return_model=return_model,
        extra_params=extra_params,
    )


def get_json_fn(extras, arch_json_partial, opt_config_partial):
    datasets = extras['datasets']
    resp_train = get_resp_train(datasets)
    assert datasets['X_train'].shape[1] == 1
    model_json = arch_json_partial(list(datasets['X_train'].shape[2:]),
                                   resp_train.shape[1])
    opt_config = opt_config_partial(model_json=model_json)

    return {
        'model_json': model_json,
        'opt_config': opt_config,
    }


def init_bias(model: JSONNet, mean_response: np.ndarray):
    # try getting bn_output's bias.
    # if not, try fc's.
    bias_module = model.get_module_optional('bn_output')
    if bias_module is None:
        bias_module = model.get_module('fc')
    else:
        assert model.get_module('fc').bias is None

    init_bias_wrapper(
        bias_module.bias,
        mean_response,
        init_type=check_init_type(model.get_module('final_act')),
    )


def add_conv_layers(model: JSONNet):
    param_dict = model.get_param_dict()
    conv_layer_names = param_dict['comments']['conv_layers']
    conv_layer_names_set = set(conv_layer_names)
    assert len(conv_layer_names) == len(conv_layer_names_set)
    conv_layers_dict = dict()
    for x, y in model.moduledict.named_modules():
        if x in conv_layer_names_set:
            conv_layers_dict[x] = y
    assert len(conv_layers_dict) == len(conv_layer_names_set)
    model.extradict['conv_layers'] = [conv_layers_dict[z] for z in
                                      conv_layer_names]


def initialize_model(model: JSONNet, extras):
    # get all conv layers, for optimizer.
    add_conv_layers(model)

    resp_train = get_resp_train(extras['datasets'])
    init_bias(model, resp_train.mean(axis=0))
