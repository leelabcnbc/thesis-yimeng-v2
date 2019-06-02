"""a very generic nested training loop

the learning has a nested loop structure.

I call the outer loop phases.

In each phase, I may set a different learning rate
and different early quitting criteria.

each phase has a max iteration, such as 5000.


Inside each phase, there is another loop that loops over epochs.

at end of every epoch, we compute training set stats.
(optionally) at end of every N_val epoch, we compute validation stats.
# we keep track of the best model epoch number. for val.
(optionally) at end of every N_test epoch, we compute testing stats.

validation stats can be used to do early stopping,
after `patience` number of iterations.

"""
# adapted from <https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal_utils/training.py>  # noqa: E501
from copy import deepcopy
from typing import Callable, Union, Any, Optional

from torch import nn, optim, Tensor
import torch
from torch.utils.data import DataLoader

import numpy as np

from .utils import AverageMeter


def fill_in_config(config: dict):
    # TODO should be verified using JSONschema... no time.
    # make a copy first
    config = deepcopy(config)
    assert config.keys() <= {'global', 'per_phase'}

    if 'global' not in config:
        config['global'] = {}
    if 'per_phase' not in config:
        config['per_phase'] = []

    template_global_dict = {
        # always first GPU. use `CUDA_VISIBLE_DEVICES` outside to select
        # the actual GPU.
        'device': 'cuda',
        'loss_every_iter': 20,  # show loss every 20 iterations,
        'val_every': 1,
        'test_every': 1,
        'early_stopping_field': 'loss',
        'show_every': 1,
        'num_input': 1,  # the rest are labels.
    }

    assert template_global_dict.keys() >= config['global'].keys()
    template_global_dict.update(config['global'])
    config['global'] = template_global_dict

    # then handling each phase
    for i in range(len(config['per_phase'])):
        template_per_phase_dict = {
            'max_epoch': 10000,
            'lr_config': None,
            'early_stopping_config': {'patience': 10},
        }
        per_phase_dict_this = config['per_phase'][i]
        assert template_per_phase_dict.keys() >= per_phase_dict_this.keys()
        template_per_phase_dict.update(per_phase_dict_this)
        lr_config = template_per_phase_dict['lr_config']
        # check the lr config part.
        if lr_config is not None:
            update_type = lr_config['type']
            if update_type == 'reduce_by_factor':
                assert lr_config.keys() == {'type', 'factor'}

            elif update_type == 'fixed':
                assert lr_config.keys() == {'type', 'fixed_lr'}
            else:
                raise NotImplementedError
        config['per_phase'][i] = template_per_phase_dict

    return config


def train(model: nn.Module, loss_func: Callable[[Any], Tensor],
          optimizer: optim.Optimizer,
          *,
          config: dict,
          f_best: str,
          dataset_train: Union[Callable, DataLoader],
          dataset_val: Optional[DataLoader] = None,
          dataset_test: Optional[DataLoader] = None,
          eval_fn: Optional[Callable[[Any], dict]] = None,
          legacy_random_seed: bool = False
          ):
    num_phase = len(config['per_phase'])

    # this is for early stopping only.
    stats_best = {
        'best_phase': None,
        'best_epoch': None,
        'early_stopping_loss': float('infinity'),
        'stats': None,
    }
    stats_all = []

    for i_phase, phase_config_dict in enumerate(config['per_phase']):
        print(f'========starting phase {i_phase + 1}/{num_phase}==========')

        stats_this_phase, stats_best = train_one_phase(model, loss_func,
                                                       dataset_train,
                                                       optimizer,
                                                       phase_config_dict,
                                                       config['global'],
                                                       dataset_val,
                                                       dataset_test,
                                                       eval_fn,
                                                       stats_best,
                                                       f_best, i_phase,
                                                       legacy_random_seed)
        stats_all.append(stats_this_phase)
        print(f'========end phase {i_phase + 1}/{num_phase}==========')

    return {
        'stats_all': stats_all,
        'stats_best': stats_best,
    }


def _update_lr(optimizer: optim.Optimizer, lr_config):
    update_type = lr_config['type']
    if update_type == 'reduce_by_factor':
        assert lr_config.keys() == {'type', 'factor'}

        def update_func(x, i):
            if np.isscalar(lr_config['factor']):
                return x * lr_config['factor']
            else:
                return x * lr_config['factor'][i]

    elif update_type == 'fixed':
        assert lr_config.keys() == {'type', 'fixed_lr'}

        def update_func(_, i):
            if np.isscalar(lr_config['fixed_lr']):
                return lr_config['fixed_lr']
            else:
                return lr_config['fixed_lr'][i]

    else:
        raise NotImplementedError

    for idx, p in enumerate(optimizer.param_groups):
        old_lr = p['lr']
        new_lr = update_func(old_lr, idx)
        print("for grp of sz {}, lr from {:.6f} to {:.6f}".format(
            len(p['params']), old_lr, new_lr))
        p['lr'] = new_lr
    return


def _load_state(f_best, model, optimizer):
    checkpoint = torch.load(f_best)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def _save_state(f_best, model, optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f_best)


def _setup_one_phase(global_config_dict, phase_config_dict):
    loss_every_iter = global_config_dict['loss_every_iter']
    val_every = global_config_dict['val_every']
    test_every = global_config_dict['test_every']
    show_every = global_config_dict['show_every']
    num_input = global_config_dict['num_input']

    max_epoch = phase_config_dict['max_epoch']
    early_stopping_config = phase_config_dict['early_stopping_config']
    lr_config = phase_config_dict['lr_config']

    es_field = global_config_dict['early_stopping_field']

    device = global_config_dict['device']

    return {
        'loss_every_iter': loss_every_iter,
        'val_every': val_every,
        'test_every': test_every,
        'show_every': show_every,
        'num_input': num_input,
        'max_epoch': max_epoch,
        'early_stopping_config': early_stopping_config,
        'lr_config': lr_config,
        'es_field': es_field,
        'device': device,
    }


def _setup_one_phase_es(conf):
    if conf['early_stopping_config'] is not None:
        early_stopping_patience = conf['early_stopping_config']['patience']
        # like keras.
        assert early_stopping_patience > 0
        early_stopping_wait = 0
        # otherwise we may leave the training without having a model.
        assert conf['val_every'] is not None and conf['val_every'] <= conf[
            'max_epoch']

        return {
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_wait': early_stopping_wait,
        }
    else:
        return None


def _get_on_device_data(data_this_batch, device, num_input):
    data_this_batch = [x.to(device) for x in data_this_batch]

    inputs = data_this_batch[:num_input]
    labels = data_this_batch[num_input:]
    return inputs, labels


def _train_one_epoch(i_epoch, model, optimizer, loss_func,
                     conf,
                     dataset_train, print_flag) -> dict:
    loss_meter = AverageMeter()

    if print_flag:
        print(f'========starting epoch {i_epoch}==========')

    # do the standard thing.
    # load dataset
    if isinstance(dataset_train, DataLoader):
        dataset = dataset_train
    else:
        dataset = dataset_train()

    for i_minibatch, data_this_batch in enumerate(dataset):
        # double check that I'm training properly.
        # in other words, I'm not doing dropout improperly.
        assert model.training
        inputs, labels = _get_on_device_data(data_this_batch,
                                             conf['device'],
                                             conf['num_input'])

        optimizer.zero_grad()

        # inputs unpacked.
        outputs = model(*inputs)

        # not unpacked
        loss = loss_func(outputs, labels, model)

        loss.backward()
        optimizer.step()

        # then let's do things.
        if print_flag and i_minibatch % conf['loss_every_iter'] == 0:
            print(f'{i_epoch}-{i_minibatch}, train loss {loss.item()}')

        # here I ignored number of items, as I assume that each batch has
        # same size;
        # also, due to some regularization terms inside loss as well,
        # it's difficult to say what `n` we should use for meter,
        # so it's simpler to stick to 1.
        loss_meter.update(loss.item())

    stats_this_batch = {'train': loss_meter.avg}

    if print_flag:
        print('train loss', loss_meter.avg)

    return stats_this_batch


def _val_test_one_epoch(i_epoch, model, eval_fn, loss_func,
                        dataset, conf, print_flag,
                        val_test_every, msg) -> Optional[dict]:
    if val_test_every is not None and i_epoch % val_test_every == 0:
        assert dataset is not None
        # then print some data for validation set
        metric = eval_wrapper(model, dataset,
                              conf['device'],
                              conf['num_input'],
                              eval_fn, loss_func)
        assert metric is not None

        if print_flag:
            print(msg, metric)
    else:
        metric = None

    return metric


def train_one_phase(model, loss_func, dataset_train,
                    optimizer: optim.Optimizer,
                    phase_config_dict, global_config_dict, dataset_val,
                    dataset_test,
                    eval_fn, stats_best, f_best, phase_idx,
                    legacy_random_seed):
    conf = _setup_one_phase(global_config_dict, phase_config_dict)
    conf_es = _setup_one_phase_es(conf)
    # for sanity check
    del global_config_dict
    del phase_config_dict

    if conf['lr_config'] is not None:
        # TODO right now we use fixed LR per phase.
        #   allow per epoch LR strategy as well.
        _update_lr(optimizer, conf['lr_config'])

    stats_this_phase = []

    if legacy_random_seed:
        # ran val once to make sure random seeds are the same
        # as in my old code.
        # this is a "bug" of PyTorch.
        #
        # <https://github.com/pytorch/pytorch/issues/11062>
        _val_test_one_epoch(0, model,
                            eval_fn,
                            loss_func,
                            dataset_val, conf,
                            True,
                            conf['val_every'],
                            'val metric init')

    model.train()

    for i_epoch in range(conf['max_epoch']):

        print_flag = i_epoch % conf['show_every'] == 0

        stats_this_batch = _train_one_epoch(i_epoch, model, optimizer,
                                            loss_func,
                                            conf,
                                            dataset_train, print_flag)

        stats_this_batch['val'] = _val_test_one_epoch(i_epoch, model,
                                                      eval_fn,
                                                      loss_func,
                                                      dataset_val, conf,
                                                      print_flag,
                                                      conf['val_every'],
                                                      'val metric')

        stats_this_batch['test'] = _val_test_one_epoch(i_epoch, model,
                                                       eval_fn,
                                                       loss_func,
                                                       dataset_test, conf,
                                                       print_flag,
                                                       conf['test_every'],
                                                       'test metric')

        if print_flag:
            print(f'========done epoch {i_epoch}==========')

        stats_this_phase.append(stats_this_batch)

        # do early stopping stuff.
        # val_metric is not None means we evaluated dataset_val this time.
        if conf_es is not None and stats_this_batch['val'] is not None:
            assert np.isfinite(
                stats_this_batch['val'][
                    conf['es_field']]), 'validation metric must be finite'
            if (stats_this_batch['val'][conf['es_field']] <
                    stats_best['early_stopping_loss']):
                conf_es['early_stopping_wait'] = 0
                stats_best = {
                    'best_phase': phase_idx,
                    'best_epoch': i_epoch,
                    'early_stopping_loss': stats_this_batch['val'][
                        conf['es_field']],
                    # this is just a copy for convenience.
                    # you can always read from corresponding phase and epoch
                    'stats': deepcopy(stats_this_batch),
                }

                # save state
                _save_state(f_best, model, optimizer)
            else:
                conf_es['early_stopping_wait'] += 1
                # print(f'patience {early_stopping_wait}')
                if (conf_es['early_stopping_wait'] >=
                        conf_es['early_stopping_patience']):
                    print(f'early stopping after epoch {i_epoch}',
                          'metric', stats_best['early_stopping_loss'])

                    _load_state(f_best, model, optimizer)
                    break
    else:
        # quit without break
        if conf_es is not None:
            # if this is the first phase in training,
            # sure we must have improved from inf
            # else
            #   if last phase comes with early stopping,
            #       early_stopping_best is not inf in the first place.
            #   else last phase does not have early stopping.
            #       early_stopping_best is inf in the beginning,
            #       and one validation test is going to make it
            #       smaller than inf

            # we typically have assume that every phase has or has no early
            # stopping.
            #
            # if you mix them, a phase without early stopping
            # would nullify all previous early stopping stats.

            assert stats_best['early_stopping_loss'] < float('infinity')
            print(f"recover best model after {conf['max_epoch']} epochs",
                  'metric', stats_best['early_stopping_loss'])
            _load_state(f_best, model, optimizer)
        else:
            # just save last one.
            _save_state(f_best, model, optimizer)
            stats_best = {
                'best_phase': phase_idx,
                'best_epoch': conf['max_epoch'] - 1,
                'early_stopping_loss': float('infinity'),
                # this is just a copy for convenience.
                # you can always read from corresponding phase and epoch
                'stats': deepcopy(stats_this_phase[-1]),
            }

    model.eval()

    return stats_this_phase, stats_best


def eval_wrapper(model: nn.Module, dataset: DataLoader,
                 device, num_input,
                 eval_fn, loss_func) -> dict:
    # some part inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py  # noqa: E501
    #
    # collect both output and target
    model.eval()
    labels_all = []
    outputs_all = []
    loss_meter = AverageMeter()

    for i_minibatch, data_this_batch in enumerate(dataset):
        inputs, labels = _get_on_device_data(data_this_batch,
                                             device,
                                             num_input)
        with torch.no_grad():
            outputs = model(*inputs)
            loss_meter.update(loss_func(outputs, labels, model).item())

        if isinstance(outputs, tuple):
            outputs = [x.cpu().numpy().copy() for x in outputs]
        else:
            outputs = outputs.cpu().numpy().copy()
        outputs_all.append(outputs)
        labels = [x.cpu().numpy().copy() for x in labels]
        labels_all.append(labels)

    stats = {'loss': loss_meter.avg}

    if eval_fn is not None:
        # just pass in raw numpy stuff, and let eval_fn itself figure
        # things out
        # this is OK as our data sets are going to be small.
        # for ImageNet, we will use another training wrapper anyway.
        stats_additional = eval_fn(outputs_all, labels_all)
    else:
        stats_additional = {}

    assert 'loss' not in stats_additional
    stats.update(stats_additional)
    model.train()
    return stats
