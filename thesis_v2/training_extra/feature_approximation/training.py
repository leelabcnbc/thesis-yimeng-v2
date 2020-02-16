from functools import partial

import torch

"""training should invoke this function"""

from .opt import get_optimizer
from .loss import get_loss

from ..training import train_one_wrapper


def train_one(*,
              arch_json_partial, opt_config_partial, datasets,
              key,
              show_every=1000,
              model_seed=0, train_seed=0,
              max_epoch=20000,
              device='cuda',
              return_model=True,
              batch_size=256,
              num_phase=3,
              ):
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
        initialize_model_fn=lambda x1, x2: None,
        get_optimizer_fn=get_optimizer,
        get_loss_fn=get_loss,
        datasets=datasets,
        key=key,
        show_every=show_every,
        model_seed=model_seed, train_seed=train_seed,
        max_epoch=max_epoch,
        early_stopping_field=None,
        device=device,
        val_test_every=None,
        return_model=return_model,
        extra_params={
            'datasets': {'y_dim': 4, 'batch_size': batch_size},
            'training_extra_config': {'num_phase': num_phase},
        },
    )


def get_json_fn(extras,
                arch_json_partial,
                opt_config_partial,
                ):
    datasets = extras['datasets']
    model_json = arch_json_partial(
        list(datasets['X_train'].shape[1:]),
        # TODO: fix the code for local pcn mask cnn
        list(datasets['y_train'].shape[1:])
    )
    opt_config = opt_config_partial(model_json=model_json)

    return {
        'model_json': model_json,
        'opt_config': opt_config,
    }

