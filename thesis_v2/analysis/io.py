import json
from os.path import join

import numpy as np
import pandas as pd

from torchnetjson.builder import build_net

from ..configs.model.maskcnn_polished_with_rcnn_k_bl import (
    keygen
)
from ..training.training_aux import load_training_results
from ..training_extra.misc import count_params
from .. import dir_dict

from ..models.maskcnn_polished_with_rcnn_k_bl.builder import load_modules

load_modules()


def collect_rcnn_k_bl_main_result(*,
                                  fixed_keys,
                                  generator,
                                  total_num_param,
                                  train_size_mapping,
                                  cc_max_all_neurons,
                                  num_neuron,
                                  ):
    rows_all = []

    param_set = None

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

        # load model to get param count
        key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})
        # 10 to go.
        result = load_training_results(key, return_model=False)
        # load twice, first time to get the model.
        result = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))
        num_param = count_params(result['model'])

        cc_native = np.asarray(result['stats_best']['stats']['test']['corr'])

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

        if cc_max_all_neurons is not None:
            assert cc_max_all_neurons.shape == (num_neuron,)

        param['train_keep'] = train_size_mapping.get(param['train_keep'], param['train_keep'])
        # add result
        row_this = {
            k: v for k, v in param.items() if k not in fixed_keys
        }
        row_this['num_param'] = num_param
        row_this['cc_raw_avg'] = cc_native.mean()
        row_this['cc2_raw_avg'] = (cc_native ** 2).mean()
        if cc_max_all_neurons is not None:
            row_this['cc2_normed_avg'] = ((cc_native / cc_max_all_neurons) ** 2).mean()

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
