import json
from os.path import join
from copy import deepcopy
from collections import Counter
from typing import List

import numpy as np
from numpy.linalg import norm
import pandas as pd

from joblib.parallel import Parallel, delayed

from torchnetjson.builder import build_net
import torch

from ..configs.model.maskcnn_polished_with_rcnn_k_bl import (
    keygen
)
from ..training.training_aux import load_training_results
from ..training_extra.misc import count_params
from .. import dir_dict

from ..models.maskcnn_polished_with_rcnn_k_bl.builder import load_modules
from ..feature_extraction.extraction import (
    augment_module,
    normalize_augment_config
)

from ..blocks.rcnn_basic_kriegeskorte.nn_modules import BLConvLayerStack

from .hal_analysis_refactor.model_orientation_tuning import (
    model_orientation_tuning_one,
    get_bars,
    get_stimuli_dict
)

from .utils import get_source_analysis_for_one_model_spec, LayerSourceAnalysis

load_modules()


def collect_rcnn_k_bl_main_result(*,
                                  fixed_keys,
                                  generator,
                                  total_num_param,
                                  train_size_mapping,
                                  cc_max_all_neurons,
                                  num_neuron,
                                  internal_dynamics_cls=None,
                                  skip_eval_json=False,
                                  no_missing_data=True,
                                  key_override=None,
                                  save_avg=True,
                                  mask=None,
                                  ):
    rows_all = []
    rows_all_param_overwrite = []

    param_set = None

    if cc_max_all_neurons is not None:
        cc_max_all_neurons = cc_max_all_neurons[mask]

    if key_override is None:
        key_override = dict()

    if internal_dynamics_cls is not None:
        assert internal_dynamics_cls > 1

    for idx, (src, param) in enumerate(generator):
        assert len(param) == total_num_param
        total_param_to_explain = len(param)

        if idx % 100 == 0:
            print(idx)

        # some parameters that won't change.
        for k_fix, v_fix in fixed_keys.items():
            assert param[k_fix] == v_fix
            total_param_to_explain -= 1

        # {'yhat_reduce_pick': 'none', 'train_keep': 1280, 'model_seed': 0,
        # act_fn': 'relu', 'loss_type': 'mse', 'out_channel': 8, 'num_layer': 2,
        # 'rcnn_bl_cls': 1,
        # 'rcnn_acc_type': 'cummean', 'ff_1st_bn_before_act': True}
        # this can be useful for mind cluster, where I typically store results in different folders,
        # suffixed by dates, such as `maskcnn_polished_with_rcnn_k_bl.20200208`
        param.update(key_override)
        # load model to get param count

        key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})
        # 10 to go.
        try:
            result = load_training_results(key, return_model=False)
            # load twice, first time to get the model.
            result = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))
            num_param = count_params(result['model'])
            cc_native = np.asarray(result['stats_best']['stats']['test']['corr'])
            if mask is not None:
                cc_native = cc_native[mask]
        except FileNotFoundError as e:
            num_param = -1
            cc_native = np.full((num_neuron,), np.nan, dtype=np.float32)
            if no_missing_data:
                raise e

        # replace 'yhat_reduce_pick' + 'rcnn_acc_type' with 'readout_type'
        readout_raw = param['yhat_reduce_pick'], param['rcnn_acc_type']
        if readout_raw == (-1, 'cummean'):
            # this should only happen for deep FF models, where this does not matter.
            assert param['rcnn_bl_cls'] == 1
            assert src == 'deep-ff'

        param['readout_type'] = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
            (-1, 'cummean'): 'legacy',
        }[readout_raw]
        if param['readout_type'] == 'legacy':
            assert src == 'deep-ff'
        else:
            #             print(src)
            assert src == param['readout_type']

        del param['yhat_reduce_pick']
        del param['rcnn_acc_type']
        total_param_to_explain -= 1

        # skip if
        # 1) internal_dynamics_cls is set
        # 2) cls is not 1 nor internal_dynamics_cls
        # note that cls=1 internal dynamics cannot be extracted this way.
        # maybe I will put it to internal_dynamics_cls+1, as a hack.
        if internal_dynamics_cls is not None:
            # skip non trivial R models unless cls==internal_dynamics_cls
            skip_this_for_internal = param['rcnn_bl_cls'] != 1 and param['rcnn_bl_cls'] != internal_dynamics_cls
            use_internal_dynamics = param['rcnn_bl_cls'] == internal_dynamics_cls
        else:
            skip_this_for_internal = False
            use_internal_dynamics = False

        if not skip_eval_json:
            # load eval json
            eval_json_file = join(dir_dict['analyses'], key, 'eval.json')
            try:
                with open(eval_json_file, 'rt', encoding='utf-8') as f_eval:
                    eval_json = json.load(f_eval)
            except FileNotFoundError as e:
                print('missing file', idx)
                raise e

            cc_native_debug = np.asarray(eval_json['native'])

            if param['rcnn_bl_cls'] != 1:
                cc_native_debug_2 = np.asarray(eval_json[param['readout_type']][str(param['rcnn_bl_cls'])])
            else:
                cc_native_debug_2 = cc_native_debug

            assert cc_native_debug.shape == cc_native.shape == (num_neuron,) == cc_native_debug_2.shape

            assert np.allclose(cc_native, cc_native_debug, atol=1e-4)
            assert np.allclose(cc_native, cc_native_debug_2, atol=1e-4)
        else:
            eval_json = None

        if cc_max_all_neurons is not None:
            assert cc_max_all_neurons.shape == (num_neuron,)

        param['train_keep'] = train_size_mapping.get(param['train_keep'], param['train_keep'])
        # add result
        row_this = {
            k: v for k, v in param.items() if k not in fixed_keys
        }
        row_this['num_param'] = num_param
        if save_avg:
            row_this['cc_raw_avg'] = cc_native.mean()
            row_this['cc2_raw_avg'] = (cc_native ** 2).mean()
            if cc_max_all_neurons is not None:
                row_this['cc2_normed_avg'] = ((cc_native / cc_max_all_neurons) ** 2).mean()
        else:
            row_this['cc_raw'] = cc_native.tolist()
            row_this['cc2_raw'] = (cc_native ** 2).tolist()
            if cc_max_all_neurons is not None:
                row_this['cc2_normed'] = ((cc_native / cc_max_all_neurons) ** 2).tolist()

        if not skip_this_for_internal:
            rows_all.append(row_this)
        else:
            rows_all_param_overwrite.append(row_this)

        if use_internal_dynamics:
            # fill in cls = 2,....,internal_dynamics_cls-1,
            # and fill cls = 1 into internal_dynamics_cls + 1
            for cls_this in range(1, internal_dynamics_cls):
                # right now, the param part is wrong. will replace later,
                # using data in `rows_all_param_overwrite`
                row_this_internal = deepcopy(row_this)
                cc_this_internal = np.asarray(eval_json[param['readout_type']][str(cls_this)])

                assert cc_this_internal.shape == (num_neuron,)
                if cls_this != 1:
                    row_this_internal['rcnn_bl_cls'] = cls_this
                else:
                    row_this_internal['rcnn_bl_cls'] = internal_dynamics_cls + 1
                row_this_internal['cc_raw_avg'] = cc_this_internal.mean()
                row_this_internal['cc2_raw_avg'] = (cc_this_internal ** 2).mean()
                if cc_max_all_neurons is not None:
                    row_this_internal['cc2_normed_avg'] = ((cc_this_internal / cc_max_all_neurons) ** 2).mean()
                rows_all.append(row_this_internal)

        if param_set is None:
            param_set = set(param.keys())
        assert param_set == param.keys()

    assert param_set is not None

    df_this = pd.DataFrame(rows_all, columns=sorted(list(rows_all[0].keys())))
    df_this = df_this.set_index(
        keys=sorted([k for k in param_set if k not in fixed_keys]),
        verify_integrity=True
    ).sort_index()

    if len(rows_all_param_overwrite) > 0:
        df_this_overwrite = pd.DataFrame(rows_all_param_overwrite,
                                         columns=sorted(list(rows_all_param_overwrite[0].keys())))
        df_this_overwrite = df_this_overwrite.set_index(
            keys=sorted([k for k in param_set if k not in fixed_keys]),
            verify_integrity=True
        ).sort_index()
        cls_loc = df_this.index.names.index('rcnn_bl_cls')
        # then overwrite.
        for tuple_this in df_this_overwrite.itertuples():
            if tuple_this[0][cls_loc] < internal_dynamics_cls:
                df_this.loc[tuple_this[0], 'num_param'] = tuple_this.num_param

        # fix cls == internal_dynamics_cls + 1
        for index_this in df_this.index:
            assert type(index_this) is tuple
            if index_this[cls_loc] == internal_dynamics_cls + 1:
                index_this_new = list(index_this)
                index_this_new[cls_loc] = 1
                index_this_new = tuple(index_this_new)
                df_this.loc[index_this, 'num_param'] = df_this.loc[index_this_new, 'num_param']

    return df_this


def get_resp_fn_hal_tuning(model, stimuli):
    module_name = 'moduledict.accumulator'
    augment_config = {
        'module_names': [module_name, ]
    }
    augment_config = normalize_augment_config(augment_config)
    callback_dict, remove_handles = augment_module(model, **augment_config)

    # no need for torch.no_grad(), as it's already handled outside
    # just for safety
    with torch.no_grad():
        model(stimuli)

    # print(len(callback_dict[module_name]))

    remove_handles()
    # last one
    return callback_dict[module_name][-1]


def get_self_weights_fn(model):
    # last layer
    weights = model.moduledict['bl_stack'].layer_list[-1].l_conv.weight

    assert weights is not None

    weights = weights.detach().numpy()

    assert weights.ndim == 4
    assert weights.shape[0] == weights.shape[1]
    self_weights = [weights[i, i, :, :].copy() for i in range(weights.shape[0])]
    return self_weights


def collect_rcnn_k_bl_hal_analysis_inner(
        *,
        idx,
        src, param,
        fixed_keys,
        total_num_param,
        train_size_mapping,
        ignore_seed_in_baseline
):
    assert len(param) == total_num_param
    total_param_to_explain = len(param)

    input_size = param['input_size']

    # some parameters that won't change.
    for k_fix, v_fix in fixed_keys.items():
        assert param[k_fix] == v_fix
        total_param_to_explain -= 1

    # {'yhat_reduce_pick': 'none', 'train_keep': 1280, 'model_seed': 0,
    # act_fn': 'relu', 'loss_type': 'mse', 'out_channel': 8, 'num_layer': 2,
    # 'rcnn_bl_cls': 1,
    # 'rcnn_acc_type': 'cummean', 'ff_1st_bn_before_act': True}

    # load model to get param count
    key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})
    # 10 to go.
    result = load_training_results(key, return_model=False)
    # load twice, first time to get the model.
    result = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))
    num_param = count_params(result['model'])
    # replace 'yhat_reduce_pick' + 'rcnn_acc_type' with 'readout_type'
    readout_raw = param['yhat_reduce_pick'], param['rcnn_acc_type']
    if readout_raw == (-1, 'cummean'):
        # this should only happen for deep FF models, where this does not matter.
        assert param['rcnn_bl_cls'] == 1
        assert src == 'deep-ff'

    param['readout_type'] = {
        ('none', 'cummean'): 'cm-avg',
        (-1, 'cummean_last'): 'cm-last',
        ('none', 'instant'): 'inst-avg',
        (-1, 'last'): 'inst-last',
        (-1, 'cummean'): 'legacy',
    }[readout_raw]
    if param['readout_type'] == 'legacy':
        assert src == 'deep-ff'
    else:
        #             print(src)
        assert src == param['readout_type']

    del param['yhat_reduce_pick']
    del param['rcnn_acc_type']
    total_param_to_explain -= 1

    param['train_keep'] = train_size_mapping.get(param['train_keep'], param['train_keep'])
    # add result
    row_this = {
        k: v for k, v in param.items() if k not in fixed_keys
    }
    row_this['num_param'] = num_param

    if param['rcnn_bl_cls'] == 1:
        row_this['hal_tuning_analysis_inverted'] = None
        row_this['hal_tuning_analysis'] = None
        row_this['hal_tuning_analysis_improved'] = None
        row_this['hal_tuning_analysis_improved_baseline'] = None
        row_this['hal_tuning_analysis_half_improved'] = None
        row_this['hal_tuning_analysis_half_improved_baseline'] = None
    else:
        row_this['hal_tuning_analysis_inverted'] = model_orientation_tuning_one(
            model=result['model'].eval(),
            get_self_weights_fn=get_self_weights_fn,
            get_resp_fn=get_resp_fn_hal_tuning,
            stimuli_dict=get_stimuli_dict(new_size=input_size, inverted=True),
            bars=get_bars(),
        )
        row_this['hal_tuning_analysis'] = model_orientation_tuning_one(
            model=result['model'].eval(),
            get_self_weights_fn=get_self_weights_fn,
            get_resp_fn=get_resp_fn_hal_tuning,
            stimuli_dict=get_stimuli_dict(new_size=input_size),
            bars=get_bars(),
        )
        row_this['hal_tuning_analysis_improved'] = model_orientation_tuning_one(
            model=result['model'].eval(),
            get_self_weights_fn=get_self_weights_fn,
            get_resp_fn=get_resp_fn_hal_tuning,
            stimuli_dict=get_stimuli_dict(new_size=input_size, also_get_inverted=True),
            bars=get_bars(legacy=False),
        )

        row_this['hal_tuning_analysis_half_improved'] = model_orientation_tuning_one(
            model=result['model'].eval(),
            get_self_weights_fn=get_self_weights_fn,
            get_resp_fn=get_resp_fn_hal_tuning,
            stimuli_dict=get_stimuli_dict(new_size=input_size),
            bars=get_bars(legacy=False),
        )
        # get some initial model
        if not ignore_seed_in_baseline:
            torch.manual_seed(row_this['model_seed'])
        else:
            torch.manual_seed(idx)
        model_random = build_net(result['config_extra']['model'])
        row_this['hal_tuning_analysis_improved_baseline'] = model_orientation_tuning_one(
            model=model_random.eval(),
            get_self_weights_fn=get_self_weights_fn,
            get_resp_fn=get_resp_fn_hal_tuning,
            stimuli_dict=get_stimuli_dict(new_size=input_size, also_get_inverted=True),
            bars=get_bars(legacy=False),
        )

        row_this['hal_tuning_analysis_half_improved_baseline'] = model_orientation_tuning_one(
            model=model_random.eval(),
            get_self_weights_fn=get_self_weights_fn,
            get_resp_fn=get_resp_fn_hal_tuning,
            stimuli_dict=get_stimuli_dict(new_size=input_size),
            bars=get_bars(legacy=False),
        )

    return row_this, set(param.keys())


def collect_rcnn_k_bl_hal_analysis(*,
                                   fixed_keys,
                                   generator,
                                   total_num_param,
                                   train_size_mapping,
                                   ignore_seed_in_baseline=False,
                                   ):
    ret_all = Parallel(n_jobs=-1, verbose=5)(
        delayed(collect_rcnn_k_bl_hal_analysis_inner)(
            idx=idx,
            src=src, param=param,
            fixed_keys=fixed_keys,
            total_num_param=total_num_param,
            train_size_mapping=train_size_mapping,
            ignore_seed_in_baseline=ignore_seed_in_baseline,
        ) for idx, (src, param) in enumerate(generator)
    )
    rows_all = [x[0] for x in ret_all]
    param_set = ret_all[0][1]
    for zzz in ret_all:
        assert zzz[1] == param_set
    assert param_set is not None

    df_this = pd.DataFrame(rows_all, columns=sorted(list(rows_all[0].keys())))
    df_this = df_this.set_index(
        keys=sorted([k for k in param_set if k not in fixed_keys]),
        verify_integrity=True
    ).sort_index()

    return df_this


def compute_average_scale_of_weight(weight: np.ndarray):
    assert isinstance(weight, np.ndarray)
    # this works for 2D array as well, where x is a scalar.
    norm_list = [norm(x.ravel()) for x in weight]
    return np.asarray(norm_list).mean().item()


def get_scale_and_conv_maps_regular(bl_stack):
    bn_list_global = bl_stack.bn_layer_list
    layer_list_global = bl_stack.layer_list
    assert len(layer_list_global) == bl_stack.n_layer
    assert len(bn_list_global) == bl_stack.n_timesteps * bl_stack.n_layer
    conv_map = dict()
    scale_map = dict()
    for layer_idx, layer_this in enumerate(layer_list_global):
        layer_idx_human = layer_idx + 1
        bn_list = [bn_list_global[t * bl_stack.n_layer + layer_idx] for t in range(bl_stack.n_timesteps)]
        if layer_this.l_conv is None:
            # conv_map[f'R{layer_idx_human}'] = None
            pass
        else:
            conv_map[f'R{layer_idx_human}'] = compute_average_scale_of_weight(layer_this.l_conv.weight.detach().numpy())

        conv_map[f'B{layer_idx_human}'] = compute_average_scale_of_weight(
            layer_this.b_conv.weight.detach().numpy()
        )
        for idx, bn_layer in enumerate(bn_list, start=1):
            scale_map[f's{layer_idx_human},{idx}'] = compute_average_scale_of_weight(
                bn_layer.weight.detach().numpy() / np.sqrt(
                    bn_layer.running_var.numpy() + bn_layer.eps
                )
            )

    for vvv in conv_map.values():
        assert vvv >= 0
    for vvvv in scale_map.values():
        assert vvvv >= 0
    conv_map['I'] = 1.0
    return {
        'conv_map': conv_map,
        'scale_map': scale_map,
    }


def get_scale_and_conv_maps_multipath(bl_stack, *, fetch_only_last_timestep):
    standard_one = get_scale_and_conv_maps_regular(bl_stack)
    # then use the internal chain info to figure out the value of extra BN layers.
    # copy code in `evaluate_multi_path`
    scale_map_old = deepcopy(set(standard_one['scale_map'].keys()))
    counter = Counter()
    # I do this just to get typing.
    multipath_source: List[LayerSourceAnalysis] = bl_stack.multipath_source
    assert len(multipath_source) == bl_stack.n_timesteps
    last_step_keep = set()
    for timestep_idx, chain_list_this in enumerate(multipath_source):
        for chain_this in chain_list_this.source_list:
            # pairs of conv and BN
            assert chain_this['conv'][0] == 'I'
            chain_this_raw = chain_this['conv'][1:]
            # create sequence. note that BN has test vs train. this is encapsulated in BN itself.
            # we don't need to handle it explicitly.
            # check the implementatinon nn.Sequential. DO NOT use Sequential directly as the Sequential's
            # train/test mode is inconsistent with BN's.
            for idx, comp in enumerate(chain_this_raw):
                # obtain the corresponding component.
                if idx % 2 == 0:
                    pass
                else:
                    # here layer, time are 0-indexed
                    mod, (layer, time) = bl_stack.obtain_bn(comp, counter, return_layer_time=True)
                    c_this = counter[layer, time]
                    if c_this > 1:
                        # first one has been counted before.
                        # c_this will be 1,2,3,..., (1-indexed)
                        k = f's{layer + 1},{time + 1},{c_this - 1}'
                        assert k not in standard_one['scale_map']
                        if not fetch_only_last_timestep or (timestep_idx == bl_stack.n_timesteps - 1):
                            standard_one['scale_map'][k] = compute_average_scale_of_weight(
                                mod.weight.detach().numpy() / np.sqrt(
                                    mod.running_var.numpy() + mod.eps
                                )
                            )
                    else:
                        # may want to remove them because they are not used in last step
                        if fetch_only_last_timestep and (timestep_idx == bl_stack.n_timesteps - 1):
                            last_step_keep.add(f's{layer + 1},{time + 1}')

    if fetch_only_last_timestep:
        # remove all other ones.
        assert scale_map_old >= last_step_keep
        for k in scale_map_old - last_step_keep:
            del standard_one['scale_map'][k]
    return standard_one


def non_multi_path_block(bl_stack):
    return not bl_stack.multi_path or (bl_stack.multi_path and not bl_stack.multi_path_separate_bn)


def get_scale_and_conv_maps_for_a_model(model, *, fetch_only_last_timestep):
    bl_stack: BLConvLayerStack = model.moduledict['bl_stack']
    if non_multi_path_block(bl_stack):
        return get_scale_and_conv_maps_regular(bl_stack)
    else:
        return get_scale_and_conv_maps_multipath(bl_stack,
                                                 fetch_only_last_timestep=fetch_only_last_timestep)


def collect_rcnn_k_bl_source_analysis(*,
                                      fixed_keys,
                                      generator,
                                      total_num_param,
                                      train_size_mapping,
                                      no_missing_data=True,
                                      key_override=None,
                                      debug=False,
                                      debug_3layer=False
                                      ):
    rows_all = []

    if key_override is None:
        key_override = dict()

    param_set = None
    counted = 0
    for idx, (src, param) in enumerate(generator):
        if debug:
            if counted >= 5:
                break
        assert len(param) == total_num_param
        total_param_to_explain = len(param)

        if idx % 100 == 0:
            print(idx)

        # some parameters that won't change.
        for k_fix, v_fix in fixed_keys.items():
            assert param[k_fix] == v_fix
            total_param_to_explain -= 1

        param.update(key_override)

        # {'yhat_reduce_pick': 'none', 'train_keep': 1280, 'model_seed': 0,
        # act_fn': 'relu', 'loss_type': 'mse', 'out_channel': 8, 'num_layer': 2,
        # 'rcnn_bl_cls': 1,
        # 'rcnn_acc_type': 'cummean', 'ff_1st_bn_before_act': True}

        # load model to get param count
        key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})

        if debug:
            if debug_3layer:
                cond = (
                        param['rcnn_bl_cls'] != 4
                        or (param['yhat_reduce_pick'], param['rcnn_acc_type']) != ('none', 'cummean')
                        or param['act_fn'] != 'relu'
                        or param['ff_1st_bn_before_act']
                        or param['loss_type'] != 'mse'
                        or param['model_seed'] != 0
                        or param['num_layer'] != 3
                        or param['out_channel'] != 32
                )
            else:
                cond = (
                        param['rcnn_bl_cls'] != 5
                        or (param['yhat_reduce_pick'], param['rcnn_acc_type']) != ('none', 'cummean')
                        or param['act_fn'] != 'relu'
                        or param['ff_1st_bn_before_act']
                        or param['loss_type'] != 'mse'
                        or param['model_seed'] != 0
                        or param['num_layer'] != 2
                        or param['out_channel'] != 32
                )
            if cond:
                continue
            else:
                counted += 1

        try:
            failed_before = False
            result = load_training_results(key, return_model=False)
            # load twice, first time to get the model.
            result = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))
            num_param = count_params(result['model'])
        except FileNotFoundError as e:
            if no_missing_data:
                raise e
            failed_before = True
            num_param = -1

        # replace 'yhat_reduce_pick' + 'rcnn_acc_type' with 'readout_type'
        readout_raw = param['yhat_reduce_pick'], param['rcnn_acc_type']
        if readout_raw == (-1, 'cummean'):
            # this should only happen for deep FF models, where this does not matter.
            assert param['rcnn_bl_cls'] == 1
            assert src == 'deep-ff'

        param['readout_type'] = {
            ('none', 'cummean'): 'cm-avg',
            (-1, 'cummean_last'): 'cm-last',
            ('none', 'instant'): 'inst-avg',
            (-1, 'last'): 'inst-last',
            (-1, 'cummean'): 'legacy',
        }[readout_raw]
        if param['readout_type'] == 'legacy':
            assert src == 'deep-ff'
        else:
            #             print(src)
            assert src == param['readout_type']

        del param['yhat_reduce_pick']
        del param['rcnn_acc_type']
        total_param_to_explain -= 1

        if debug:
            print(key)

        param['train_keep'] = train_size_mapping.get(param['train_keep'], param['train_keep'])
        # add result
        row_this = {
            k: v for k, v in param.items() if k not in fixed_keys
        }
        row_this['num_param'] = num_param

        if not failed_before:
            maps = get_scale_and_conv_maps_for_a_model(
                result['model'],
                fetch_only_last_timestep=(
                        (not non_multi_path_block(result['model'].moduledict['bl_stack']))
                        and param['readout_type'] == 'inst-last'
                )
            )
            if debug:
                print(maps)
            src_analysis_instance = get_source_analysis_for_one_model_spec(
                num_recurrent_layer=param['num_layer'] - 1, num_cls=param['rcnn_bl_cls'],
                readout_type=param['readout_type'],
                separate_bn=(not non_multi_path_block(result['model'].moduledict['bl_stack'])),
            )

            if debug:
                print(src_analysis_instance.source_list)
            # get the scales for everything of this model
            row_this['source_analysis'] = src_analysis_instance.evaluate(
                scale_map=maps['scale_map'], conv_map=maps['conv_map'],
                check_all_used=True
            )
            # remove useless path
            if hasattr(result['model'].moduledict['bl_stack'], 'allowed_depth'):
                allowed_depth = result['model'].moduledict['bl_stack'].allowed_depth
                row_this['source_analysis'] = {
                    k: v for k, v in row_this['source_analysis'].items() if len(k)-1 in allowed_depth
                }
        else:
            row_this['source_analysis'] = None

        rows_all.append(row_this)

        if param_set is None:
            param_set = set(param.keys())
        assert param_set == param.keys()

    assert param_set is not None

    df_this = pd.DataFrame(rows_all, columns=sorted(list(rows_all[0].keys())))
    df_this = df_this.set_index(
        keys=sorted([k for k in param_set if k not in fixed_keys]),
        verify_integrity=True
    ).sort_index()

    return df_this
