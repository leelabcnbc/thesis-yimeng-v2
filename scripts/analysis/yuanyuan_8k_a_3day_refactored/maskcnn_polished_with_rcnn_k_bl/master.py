from typing import Optional
import json
from os.path import join, exists
from os import makedirs
import numpy as np
from functools import partial
from torchnetjson.builder import build_net
from thesis_v2 import dir_dict
from thesis_v2.data.prepared.yuanyuan_8k import get_data

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config

from thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder import (
    gen_maskcnn_polished_with_rcnn_k_bl, load_modules
)

from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    keygen
)

from thesis_v2.training_extra.data import generate_datasets
from thesis_v2.training.training_aux import load_training_results
from thesis_v2.training_extra.evaluation import eval_fn_wrapper as eval_fn_wrapper_neural
from thesis_v2.training.training import eval_wrapper

from torch.backends import cudnn
import torch

cudnn.enabled = True
cudnn.deterministic = True
cudnn.benchmark = False

load_modules()


def evaluate_one_model_using_one_strategy(
        *,
        datasets_test, model, loss_type,
        yhat_reduce_pick,
):
    # set model param
    result_on_the_go = eval_wrapper(model,
                                    datasets_test,
                                    'cuda',
                                    1,
                                    partial(eval_fn_wrapper_neural, loss_type=loss_type, yhat_reduce_axis=1,
                                            yhat_reduce_pick=yhat_reduce_pick),
                                    (lambda dummy1, dummy2, dummy3: torch.tensor(0.0)),
                                    return_responses=False,
                                    )
    return result_on_the_go['corr']


def master(*,
           split_seed,
           model_seed,
           act_fn,
           loss_type,
           input_size,
           out_channel,
           num_layer,
           kernel_size_l1,
           pooling_ksize,
           scale, scale_name,
           smoothness, smoothness_name,
           pooling_type,
           bn_after_fc,
           rcnn_bl_cls: int,
           rcnn_bl_psize: int,
           rcnn_bl_ptype: Optional[str],
           rcnn_acc_type: str,
           ff_1st_block: bool,
           ff_1st_bn_before_act: bool,
           kernel_size_l23: int,
           train_keep: int,
           model_prefix: str,
           yhat_reduce_pick: int,
           dataset_prefix: str,
           accumulator_mode: str = 'cummean',
           ):
    assert yhat_reduce_pick in {-1, 'none'}

    readout_mode_prefix = {
        'cummean': 'cm',
        'instant': 'inst',
    }[accumulator_mode]

    key = keygen(
        split_seed=split_seed,
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        input_size=input_size,
        out_channel=out_channel,
        num_layer=num_layer,
        kernel_size_l1=kernel_size_l1,
        kernel_size_l23=kernel_size_l23,
        pooling_ksize=pooling_ksize,
        scale_name=scale_name,
        smoothness_name=smoothness_name,
        pooling_type=pooling_type,
        bn_after_fc=bn_after_fc,
        rcnn_bl_cls=rcnn_bl_cls,
        rcnn_bl_psize=rcnn_bl_psize,
        rcnn_bl_ptype=rcnn_bl_ptype,
        rcnn_acc_type=rcnn_acc_type,

        ff_1st_block=ff_1st_block,
        ff_1st_bn_before_act=ff_1st_bn_before_act,
        train_keep=train_keep,
        model_prefix=model_prefix,
        yhat_reduce_pick=yhat_reduce_pick,
        dataset_prefix=dataset_prefix,
    )

    print('key', key)

    datasets = get_data('a', 200, input_size, ('042318', '043018', '051018'), scale=0.5, seed=split_seed)

    if train_keep is not None:
        assert train_keep <= 8000 * 0.8 * 0.8
        train_keep_slice = slice(train_keep)
    else:
        train_keep_slice = slice(None)

    datasets = {
        'X_train': datasets[0][train_keep_slice].astype(np.float32),
        'y_train': datasets[1][train_keep_slice],
        'X_val': datasets[2].astype(np.float32),
        'y_val': datasets[3],
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }

    def gen_cnn_partial(input_size_cnn, n):
        return gen_maskcnn_polished_with_rcnn_k_bl(
            input_size=input_size_cnn,
            num_neuron=n,
            out_channel=out_channel,
            kernel_size_l1=kernel_size_l1,  # (try 5,9,13)
            kernel_size_l23=kernel_size_l23,
            act_fn=act_fn,
            pooling_ksize=pooling_ksize,  # (try, 1,3,5,7)
            pooling_type=pooling_type,  # try (avg, max)  # looks that max works well here?
            num_layer=num_layer,
            bn_after_fc=bn_after_fc,
            n_timesteps=rcnn_bl_cls,
            blstack_pool_ksize=rcnn_bl_psize,
            blstack_pool_type=rcnn_bl_ptype,
            acc_mode=rcnn_acc_type,
            factored_constraint=None,
            ff_1st_block=ff_1st_block,
            ff_1st_bn_before_act=ff_1st_bn_before_act,
            num_input_channel=1,
        )

    opt_config_partial = partial(
        get_maskcnn_v1_opt_config,
        scale=scale,
        smoothness=smoothness,
        group=0.0,
        loss_type=loss_type,
    )

    model_dir = join(dir_dict['models'], key)

    with open(join(model_dir, 'config_extra.json'), 'rt', encoding='utf-8') as f_config:
        config_extra = json.load(f_config)

    model_json_debug = gen_cnn_partial(
        datasets['X_train'].shape[2:],
        datasets['y_train'].shape[1],
    )
    opt_json_debug = opt_config_partial(model_json=model_json_debug)
    assert model_json_debug == config_extra['model']
    assert opt_json_debug == config_extra['optimizer']

    # get cnn_partial json and opt_config_partial json, compare against existing record.

    # check if a file called `eval_done` exists
    analysis_dir = join(dir_dict['analyses'], key)
    eval_marker = join(analysis_dir, 'eval_done')
    eval_file = join(analysis_dir, 'eval.json')
    if not exists(eval_marker):
        # create dir
        makedirs(analysis_dir, exist_ok=True)

        # load dataset
        # only the test one is needed.
        datasets_test = generate_datasets(
            **datasets,
            per_epoch_train=False,
            shuffle_train=False,
        )['test']

        # load model
        result = load_training_results(key, return_model=False)
        # load twice, first time to get the model.
        result = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))
        model = result['model']
        model.cuda()
        model.eval()

        metrics_different_eval_schemes = dict()
        # then evaluate "native" mode
        metrics_different_eval_schemes['native'] = evaluate_one_model_using_one_strategy(
            datasets_test=datasets_test,
            model=model, loss_type=loss_type,
            yhat_reduce_pick=yhat_reduce_pick,
        )

        # if cycle > 1 then do the following.
        if rcnn_bl_cls > 1:
            # always set cummean = `cummean`. NOT cummean_last.
            model.moduledict['accumulator'].mode = accumulator_mode
            # for cycle = 1,2,....,T
            metrics_different_eval_schemes[f'{readout_mode_prefix}-last'] = dict()
            metrics_different_eval_schemes[f'{readout_mode_prefix}-avg'] = dict()
            assert model.moduledict['bl_stack'].n_timesteps == rcnn_bl_cls
            for bl_cls in range(1, rcnn_bl_cls + 1):
                # set model to have bl_cls cycles.
                model.moduledict['bl_stack'].n_timesteps = bl_cls
                corr_cm_last = evaluate_one_model_using_one_strategy(
                    datasets_test=datasets_test,
                    model=model, loss_type=loss_type,
                    yhat_reduce_pick=-1,
                )
                corr_cm_avg = evaluate_one_model_using_one_strategy(
                    datasets_test=datasets_test,
                    model=model, loss_type=loss_type,
                    yhat_reduce_pick='none',
                )
                # should the same as corr_cm_avg
                corr_cm_avg_debug = evaluate_one_model_using_one_strategy(
                    datasets_test=datasets_test,
                    model=model, loss_type=loss_type,
                    yhat_reduce_pick='avg',
                )
                if corr_cm_avg_debug != corr_cm_avg:
                    print(corr_cm_avg_debug, corr_cm_avg_debug,
                          np.asarray(corr_cm_avg_debug)-np.asarray(corr_cm_avg_debug))
                assert corr_cm_avg_debug == corr_cm_avg

                metrics_different_eval_schemes[f'{readout_mode_prefix}-last'][str(bl_cls)] = corr_cm_last
                metrics_different_eval_schemes[f'{readout_mode_prefix}-avg'][str(bl_cls)] = corr_cm_avg
                #
                #     evaluate at yhat_reduce_pick = -1
                #     evaluate at yhat_reduce_pick = 'none'
                #     for yhat_reduce_pick = -1, store in metrics_different_eval_schemes['cm-last'][str(cycle)]
                #     for yhat_reduce_pick = 'none', store in metrics_different_eval_schemes['cm-avg'][str(cycle)]

        # save it to JSON called `analysis_eval.json`
        with open(eval_file, 'wt', encoding='utf-8') as f_out:
            json.dump(metrics_different_eval_schemes, f_out)

        # write marker
        with open(eval_marker, 'wt', encoding='utf-8'):
            pass

        del metrics_different_eval_schemes
    # read `eval.json` and print.
    with open(eval_file, 'rt', encoding='utf-8') as f_out_read:
        metrics_different_eval_schemes_out = json.load(f_out_read)
    print(metrics_different_eval_schemes_out)
    # done.
    # result = train_one(
    #     arch_json_partial=gen_cnn_partial,
    #     opt_config_partial=opt_config_partial,
    #     datasets=datasets,
    #     key=key,
    #     max_epoch=40000,
    #     model_seed=model_seed,
    #     return_model=False,
    #     extra_params={
    #         # reduce on batch axis
    #         'eval_fn': {
    #             'yhat_reduce_axis': 1,
    #             'yhat_reduce_pick': yhat_reduce_pick,
    #         }
    #     },
    # )

    # print(result['stats_best']['stats']['test']['corr_mean'])
