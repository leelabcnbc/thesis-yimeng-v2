"""a wrapper around the main `training.py`

it mostly does some bookkeeping.
"""
from datetime import datetime
from os.path import exists
import os
import json
from typing import Optional, Callable, Union, Any
from tempfile import TemporaryDirectory
from shutil import move

from torch import nn, load, optim, Tensor
from torch.utils.data import DataLoader

from .. import dir_dict, join
from .training import train, fill_in_config

json_mapping = {
    'stats_best': 'stats_best.json',
    'stats_all': 'stats_all.json',
    'config': 'config.json',
    'config_extra': 'config_extra.json',
}


def cycle_reboot(iterable):
    # adapted from https://docs.python.org/3/library/itertools.html#itertools.cycle  # noqa: E501
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    while True:
        for element in iterable:
            yield element


def infinite_n_batch_loader(loader: DataLoader, n=1):
    # returns a function, that every time you call generates a new function
    # with n batches
    # I think to be safe, each loader should only be used once
    # with such a function.
    loader = iter(cycle_reboot(
        loader))  # ok. this has converted the loader into an infinite one.

    # if loader has shuffle=True, then it will give different things forever.

    def generate_new_n_batch():
        for _ in range(n):
            next_batch = next(loader)
            yield next_batch

    return generate_new_n_batch


def load_training_results(key: str, return_model, model=None,
                          return_checkpoint=False):
    # this can be used independently to load back a model or only
    # getting statistics.
    store_dir = join(dir_dict['models'], key)
    # get model
    if return_model:
        assert model is not None
        model.cpu()
        # only put it to CPU.
        # map_location='cpu' will prevent loading first to gpu
        # and then back to cpu (load.save always preserves device tags)

        # loading first to gpu would fail for machines without GPUs.
        # `best.pth` would store both model and optimizer states.
        model.load_state_dict(load(join(store_dir, 'best.pth'),
                                   map_location='cpu')['model'])
        model.eval()
    else:
        model = None

    result = {
        'model': model,
        'store_dir': store_dir,
        'checkpoint': load(join(store_dir,
                                'best.pth'),
                           map_location='cpu') if return_checkpoint else None,
    }

    for k, v in json_mapping.items():
        with open(join(store_dir, v), 'r', encoding='utf-8') as f_json:
            result[k] = json.load(f_json)

    return result


def training_wrapper(model: nn.Module, *,
                     loss_fn: Callable[[Any], Tensor],
                     optimizer: optim.Optimizer,
                     dataset_train: Union[DataLoader, Callable],
                     dataset_val: Optional[DataLoader],
                     dataset_test: Optional[DataLoader],
                     config: dict,
                     # for None, we just replace with {}, to make everything
                     # a dict.
                     config_extra: Optional[dict],
                     eval_fn: Optional[Callable[[Any], dict]] = None,
                     key=None,
                     return_model=True,
                     deterministic=True,
                     legacy_random_seed=False
                     ):
    # always use deterministic version
    from torch.backends import cudnn
    cudnn.enabled = True
    cudnn.deterministic = deterministic
    cudnn.benchmark = False

    # https://pytorch.org/docs/1.0.0/notes/randomness.html
    # https://github.com/pytorch/pytorch/issues/6351

    """
    we will always find a key string to save stuffs.
    if key is None, the key will be generated from UTC timestamp.
    config_extra stores the way I generate this model, opt parameters, etc.

    under join('results/models', key) (key can have `/`s in it)
    there is a `model_best.pth`, `config_extra.json`, `config.json`
    `stats_best.json`

    `config.json` should be a filled with default values from `training.py`

    `stats_all.json` that keeps track of what's going on
    during training.

    :param model: some pytorch model on CPU. eval or train not matter
                  we assume that model initialization has been done,
                  such as setting bias terms to match data set statistics.
    :param loss_fn: takes `model(input)` as input and generates a scalar
    :param optimizer: a torch.optim.Optimizer
    :param config:
    :param dataset_train:
    :param dataset_val:
    :param dataset_test:
    :param config_extra:
    :param eval_fn:
    :param key:
    :param return_model:
    :return: the model (in eval mode, on CPU), and best stats as a dict.
    """
    ts = datetime.utcnow().isoformat(timespec='microseconds')
    if key is None:
        # generate timestamp for non-keyed experimeents
        # remove ':', which is tricky to handle in many scenarios.
        key = ts.replace(':', '')
        # a bit safer... whatever.
        assert not exists(join(dir_dict['models'], key))

    config = fill_in_config(config)
    if config_extra is None:
        config_extra = {}

    # check that configs are good.
    config_normed = json.loads(json.dumps(config))
    assert config_normed == config
    config_extra_normed = json.loads(json.dumps(config_extra))
    assert config_extra_normed == config_extra

    # if the key is there, we skip training.
    store_dir = join(dir_dict['models'], key)
    if not exists(store_dir):
        os.makedirs(store_dir)
        # save timestamp
        with open(join(store_dir, 'timestamp'), 'w',
                  encoding='utf-8') as f_ts:
            f_ts.write(ts)
        # store config
        for k_config, v_config in {
            'config': config_normed,
            'config_extra': config_extra_normed
        }.items():
            with open(join(store_dir, json_mapping[k_config]),
                      'w', encoding='utf-8') as f_json_config:
                json.dump(v_config, f_json_config)

        with TemporaryDirectory(
                dir=os.environ.get('THESIS_TEMP_FOLDER', None)
        ) as temp_dir:
            f_best_tmp = join(temp_dir, 'best.pth')
            result = train(model, loss_func=loss_fn,
                           optimizer=optimizer,
                           dataset_train=dataset_train,
                           dataset_val=dataset_val,
                           dataset_test=dataset_test,
                           eval_fn=eval_fn, config=config_normed,
                           f_best=f_best_tmp,
                           legacy_random_seed=legacy_random_seed)
            # then copy/move that file
            move(f_best_tmp, join(store_dir, 'best.pth'))

        for k_stats in ('stats_all', 'stats_best'):
            v_stats = json.loads(json.dumps(result[k_stats]))
            assert v_stats == result[k_stats]
            with open(join(store_dir, json_mapping[k_stats]),
                      'w', encoding='utf-8') as f_json_stats:
                json.dump(v_stats, f_json_stats)

        ts_done = datetime.utcnow().isoformat(timespec='microseconds')
        with open(join(store_dir, 'timestamp_done'), 'w',
                  encoding='utf-8') as f_ts_done:
            f_ts_done.write(ts_done)
    else:
        # check everything exists.
        for fname in list(json_mapping.values()) + ['best.pth', ]:
            assert exists(join(store_dir, fname)), 'broken training result!'

    return load_training_results(key, return_model, model)
