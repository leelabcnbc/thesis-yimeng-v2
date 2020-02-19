from functools import partial

import torch

from torchnetjson.builder import build_net

"""training should invoke this function"""

from ..training.training_aux import training_wrapper

from .misc import count_params
from .data import generate_datasets
from .evaluation import eval_fn_wrapper

from .config import get_config


def train_one_inner(*,
                    model, datasets, key, optimizer, loss_fn,
                    config_extra,
                    seed,
                    config,
                    eval_fn,
                    return_model,
                    extra_params,
                    shuffle_train=True,
                    ):
    # does three things.
    # 1. seed
    # 2. get dataloaders
    # 3. train

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    assert isinstance(datasets, dict)
    datasets_done = generate_datasets(
        **datasets,
        per_epoch_train=True, shuffle_train=shuffle_train,
        **extra_params.get('datasets', {}),
    )

    return training_wrapper(model,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            dataset_train=datasets_done['train'],
                            dataset_val=datasets_done['val'],
                            dataset_test=datasets_done['test'],
                            config_extra=config_extra,
                            config=config,
                            eval_fn=eval_fn,
                            key=key, return_model=return_model,
                            legacy_random_seed=True)


def train_one_wrapper(*,
                      get_json_fn,
                      initialize_model_fn,
                      get_optimizer_fn,
                      get_loss_fn,
                      datasets,
                      key,
                      show_every,
                      model_seed, train_seed,
                      max_epoch,
                      early_stopping_field,
                      device,
                      val_test_every,
                      return_model,
                      extra_params=None,
                      print_model=False,
                      ):
    if extra_params is None:
        extra_params = dict()

    assert device is not None
    if model_seed is not None:
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
    # initialize
    # arch_json_partial is a function with two args.
    # first being input shape
    # second being number of neurons.

    # get all jsons ready (model_json, opt_config)
    json_dict = get_json_fn({'datasets': datasets})
    model_json = json_dict['model_json']
    opt_config = json_dict['opt_config']
    # initialize the model
    model = build_net(model_json)
    initialize_model_fn(model, {'datasets': datasets})

    print('num_param', count_params(model))
    model = model.to(device)
    model = model.train()
    if print_model:
        print(model)

    loss_fn = get_loss_fn(opt_config=opt_config)
    optimizer = get_optimizer_fn(model, opt_config['optimizer'])

    # these two can be found in `init.pth` of a trained model.
    if 'init_model_state_dict' in extra_params:
        model.load_state_dict(extra_params['init_model_state_dict'])

    if 'init_optimizer_state_dict' in extra_params:
        optimizer.load_state_dict(extra_params['init_optimizer_state_dict'])

    config = get_config(
        device=device,
        max_epoch=max_epoch,
        val_test_every=val_test_every,
        early_stopping_field=early_stopping_field,
        show_every=show_every,
        **extra_params.get('training_extra_config', {}),
    )

    eval_fn = partial(eval_fn_wrapper, loss_type=opt_config['loss'], **extra_params.get('eval_fn', {}))

    return train_one_inner(
        model=model,
        datasets=datasets,
        seed=train_seed,
        eval_fn=eval_fn,
        key=key,
        return_model=return_model,
        config_extra={'optimizer': opt_config,
                      'model': model_json},
        config=config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        extra_params=extra_params,
    )
