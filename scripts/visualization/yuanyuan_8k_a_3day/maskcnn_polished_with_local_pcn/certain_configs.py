#!/usr/bin/env python
# coding: utf-8

# this notebook tries to visualize some models examined in `basic_for_certain_configs_cc2normed.ipynb`. I will check those with biggest ccnorm2 difference between unroll=1 and unroll=5. I will ignore the middle ones, and see hope to see some visible differences.
# 
# the models are exported using `scripts/model_export/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn/export_certain_configs.py` and verified with `scripts/model_export/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn/export_certain_configs_check_keras.py`.
# 
# the visualization follows `https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/results_ipynb/plots/main/demo_neuron_vis.ipynb`

from sys import path, argv
from collections import defaultdict
from os.path import join, dirname, exists
from os import makedirs
from thesis_v2 import dir_dict
import numpy as np
from skimage import img_as_uint
from skimage.io import imsave
from pickle import dump, load

folder_to_check = 'scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn'
path.insert(0, join(dir_dict['root'], folder_to_check))
from submit_certain_configs import param_iterator_obj
from key_utils import keygen, script_keygen

from keras import backend as K

# this may not be necessary as onnx2keras handles this as well. but just to be double safe.
K.set_image_data_format('channels_first')
import onnx
from onnx2keras import onnx_to_keras

from vis.visualization import visualize_activation
from vis.utils import utils

# In[3]:
# compute ccmax
from strflab.stats import cc_max
from thesis_v2.data.prepared.yuanyuan_8k import get_neural_data_per_trial

# In[4]:


# then let's check the avg(ccnorm2 at unroll=5) - avg(ccnorm2 at unroll=1) for all cases.


# In[5]:


from thesis_v2.training.training_aux import load_training_results

constant_params = {
    'out_channel': 16,
    'num_layer': 2,
    'kernel_size_l1': 9,
    'pooling_ksize': 3,
    'pooling_type': 'avg',
    'split_seed': 'legacy',
    'smoothness_name': '0.000005',
    'scale_name': '0.01',
    'pcn_no_act': False,
    'pcn_bias': True,
    'pcn_bypass': False,
    'bn_after_fc': False,
}


def do_one_case(param, dict_ret, dict_per_ret, total_score_sum_ret, cc_max_all_neurons):
    for k1, v1 in constant_params.items():
        assert param[k1] == v1

    # assert param['model_seed'] == 0

    key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})

    params_variable = {
        k: v for k, v in param.items() if k not in (constant_params.keys() | {'scale', 'smoothness'})
    }
    # 128 (configs; 7**2) * 3 (seeds) * 6 (number of cycles from 0-5)
    assert len(params_variable) == 9
    #     print(params_variable)

    key_for_result_dict = frozenset({(k, v) for k, v in params_variable.items() if k not in {'model_seed', 'pcn_cls'}})
    assert len(key_for_result_dict) == 7

    # load the performance.
    result = load_training_results(key, return_model=False)

    if key_for_result_dict not in dict_ret:
        assert key_for_result_dict not in dict_per_ret
        dict_ret[key_for_result_dict] = defaultdict(list)
        dict_per_ret[key_for_result_dict] = defaultdict(list)
    else:
        assert key_for_result_dict in dict_per_ret

    # calculate ccnorm^2.
    cc_raw = np.asarray(result['stats_best']['stats']['test']['corr'])
    assert cc_raw.shape == (79,)

    ccnorm2_per = (cc_raw / cc_max_all_neurons) ** 2

    ccnorm2 = ccnorm2_per.mean()
    assert (np.isfinite(ccnorm2))

    total_score_sum_ret += ccnorm2_per

    dict_ret[key_for_result_dict][params_variable['pcn_cls']].append(ccnorm2)

    dict_per_ret[key_for_result_dict][params_variable['pcn_cls']].append(ccnorm2_per)


# In[6]:


def collect_data():
    cc_max_all_neurons = cc_max(get_neural_data_per_trial(
        ('042318', '043018', '051018',))
    )
    assert cc_max_all_neurons.shape == (79,)

    dict_ret = dict()
    dict_per_ret = dict()

    # check which neurons get explained best.
    total_score_sum_ret = np.zeros((79,), dtype=np.float64)

    model_count = len(list(param_iterator_obj.generate()))
    # this should be 2304, which is 128 (configs) * 3 (seeds) * 6 (number of cycles from 0-5)
    assert model_count == 2304
    for idx, param_dict in enumerate(param_iterator_obj.generate()):
        if idx % 100 == 0:
            print(f'model {idx}/{model_count}')
        do_one_case(param_dict, dict_ret, dict_per_ret, total_score_sum_ret, cc_max_all_neurons)
    return {
        'dict_all_ccnorm2': dict_ret,
        'dict_all_ccnorm2_per': dict_per_ret,
        'total_score_sum': total_score_sum_ret,
        'cc_max_all_neurons': cc_max_all_neurons,
    }


# In[10]:


def show_diff_every_key(dict_ret, dict_per_ret, neurons_to_check_this_one_dict_ret, global_vars):
    neurons_to_check_global = global_vars['neurons_to_check_global']
    cc_max_all_neurons = global_vars['cc_max_all_neurons']
    for key, value in dict_ret.items():
        if np.mean(value[5]) - np.mean(value[1]) >= 0.02:
            print(tuple(key), np.mean(value[5]) - np.mean(value[1]), np.mean(value[5]) - np.mean(value[0]),
                  np.mean(value[5]), np.mean(value[0]))
            # per neuron 
            ccnorm2_5 = np.asarray(dict_per_ret[key][5]).mean(axis=0)
            ccnorm2_1 = np.asarray(dict_per_ret[key][1]).mean(axis=0)
            ccnorm2_0 = np.asarray(dict_per_ret[key][0]).mean(axis=0)

            assert ccnorm2_5.shape == ccnorm2_1.shape == ccnorm2_0.shape == (79,)
            print(neurons_to_check_global, cc_max_all_neurons[neurons_to_check_global])
            print(5, 'global', ccnorm2_5[neurons_to_check_global])
            print(1, 'global', ccnorm2_1[neurons_to_check_global])
            print(0, 'global', ccnorm2_0[neurons_to_check_global])

            diff_5_1 = ccnorm2_5 - ccnorm2_1
            diff_5_0 = ccnorm2_5 - ccnorm2_0

            neurons_to_check_this_one = np.argsort(diff_5_1 + diff_5_0)[::-1][:5]

            print(neurons_to_check_this_one, cc_max_all_neurons[neurons_to_check_this_one])
            print(5, 'this', ccnorm2_5[neurons_to_check_this_one])
            print(1, 'this', ccnorm2_1[neurons_to_check_this_one])
            print(0, 'this', ccnorm2_0[neurons_to_check_this_one])

            print('\n' * 5)

            neurons_to_check_this_one_dict_ret[key] = neurons_to_check_this_one.tolist()


# In[11]:

def get_candidates_to_visualize():
    # in these cases, the differences look big enough.
    candidates_to_visualize = [
            # 0.062192652728396414 0.03907294184369914 0.5513841263769397 0.5123111845332405
            (('loss_type', 'poisson'), ('input_size', 50), ('pcn_bn', False), ('pcn_final_act', True), ('pcn_bn_post', False), ('act_fn', 'relu'), ('bn_before_act', True)),
            # 0.0629323663988261 0.016065903548365235 0.5353156029314999 0.5192496993831347
            (('pcn_final_act', False), ('input_size', 50), ('pcn_bn', False), ('loss_type', 'mse'), ('pcn_bn_post', False), ('bn_before_act', True), ('act_fn', 'softplus')),
            # 0.04289359964839312 0.017826213557316284 0.581971909255969 0.5641456956986527
            (('loss_type', 'poisson'), ('pcn_final_act', False), ('input_size', 50), ('pcn_bn', False), ('pcn_bn_post', False), ('bn_before_act', True), ('act_fn', 'softplus')),
        # 0.026568696278369885 0.034808706083833796 0.6656576629735268 0.630848956889693
        (('loss_type', 'poisson'), ('pcn_bn', False), ('pcn_final_act', True), ('input_size', 100),
         ('bn_before_act', True),
         ('pcn_bn_post', True), ('act_fn', 'softplus')),
    ]
    return candidates_to_visualize


# In[15]:

def visualize_one_case_inner_inner(
        *,
        model_this,
        neurons_to_check,
        seed,
        tv_weight,
        lp_norm_weight,
):
    layer_idx = utils.find_layer_idx(model_this, 'output')

    images_all = []
    for filter_idx in neurons_to_check:
        np.random.seed(seed)
        img = visualize_activation(model_this, layer_idx, filter_indices=filter_idx,
                                   tv_weight=tv_weight, lp_norm_weight=lp_norm_weight,
                                   input_range=(0.0, 255.0))
        h, w, c = img.shape
        assert h == w and c == 1
        img = np.broadcast_to(img, shape=(h, w, 3))
        images_all.append(img)
    img_final = utils.stitch_images(images_all, cols=5, margin=5)
    img_final = np.pad(img_final, [(5, 5), (5, 5), (0, 0)], mode='constant')
    print(img_final.min(), img_final.max())

    # this is float in [0.0, 255.0].s
    return img_final


def visualize_one_case_inner(*,
                             model_this,
                             neurons_to_check_this_one,
                             seed,
                             tv_weight,
                             lp_norm_weight,
                             global_vars,
                             ):
    ret_dict = dict()

    ret_dict['global'] = visualize_one_case_inner_inner(
        model_this=model_this,
        neurons_to_check=global_vars['neurons_to_check_global'],
        seed=seed,
        tv_weight=tv_weight,
        lp_norm_weight=lp_norm_weight,
    )

    ret_dict['this'] = visualize_one_case_inner_inner(
        model_this=model_this,
        neurons_to_check=neurons_to_check_this_one,
        seed=seed,
        tv_weight=tv_weight,
        lp_norm_weight=lp_norm_weight,
    )

    return ret_dict


def visualize_one_case(
        *,
        key_for_result_dict,
        tv_weight_str,
        lp_norm_weight_str,
        seed,
        global_vars,
):
    # show cls=0, 1, 5
    neurons_to_check_this_one = global_vars['neurons_to_check_this_one_dict'][key_for_result_dict]
    print(tuple(key_for_result_dict), neurons_to_check_this_one)
    for cls_to_check in (0, 1, 5):
        # for each one, show model_seed=0,1,2
        for model_seed in range(3):
            param_dict = {
                **constant_params, **{k: v for (k, v) in key_for_result_dict},
                **{'pcn_cls': cls_to_check, 'model_seed': model_seed}
            }
            print(f'cls {cls_to_check}, seed {model_seed}')
            key = keygen(**param_dict)
            img_keys = {'global', 'this'}
            img_full_dict = {
                img_name: join(dir_dict['visualization'],
                               key,
                               f'visseed{seed}',
                               f'tv{tv_weight_str}',
                               f'lp{lp_norm_weight_str}',
                               img_name + '.png')
                for img_name in img_keys
            }
            # do the visualization.
            do_something = False
            for img_path in img_full_dict.values():
                if not exists(img_path):
                    do_something = True

            if not do_something:
                continue

            # load model
            model_loc = join(dir_dict['models'], 'onnx_export', key, 'model.onnx')
            k_model = onnx_to_keras(onnx.load(model_loc), ['input'], verbose=False)

            ret_dict = visualize_one_case_inner(
                model_this=k_model,
                neurons_to_check_this_one=neurons_to_check_this_one,
                seed=seed,
                tv_weight=float(tv_weight_str),
                lp_norm_weight=float(lp_norm_weight_str),
                global_vars=global_vars,
            )
            assert ret_dict.keys() == img_keys

            # save plots
            for img_name, img in ret_dict.items():
                img_name_full = img_full_dict[img_name]
                makedirs(dirname(img_name_full), exist_ok=True)
                imsave(img_name_full, img_as_uint(img / 255))


def visualize_all_cases(*,
                        tv_weight_str,
                        lp_norm_weight_str,
                        seed,
                        global_vars,
                        ):
    for cand in global_vars['candidates_to_visualize']:
        visualize_one_case(key_for_result_dict=frozenset(cand),
                           tv_weight_str=tv_weight_str,
                           lp_norm_weight_str=lp_norm_weight_str,
                           seed=seed,
                           global_vars=global_vars,
                           )


def main():
    # while I set a seed using `seed_str`, somehow keras-vis won't give identical results across runs,
    # probably due to some undeterminism in tensorflow.
    tv_weight_str, lp_norm_weight_str, seed_str = argv[1:]

    data_pkl = join(dir_dict['visualization'],
                    'yuanyuan_8k_a_3day',
                    'maskcnn_polished_with_local_pcn',
                    'certain_configs', 'data.pkl')

    if not exists(data_pkl):
        ddd = collect_data()
        assert len(ddd) == 4
        makedirs(dirname(data_pkl), exist_ok=True)
        with open(data_pkl, 'wb') as f_data:
            dump(
                ddd,
                f_data
            )
    with open(data_pkl, 'rb') as f_data:
        data_ddd = load(f_data)
    global_vars = dict()
    total_score_sum = data_ddd['total_score_sum']
    dict_all_ccnorm2 = data_ddd['dict_all_ccnorm2']
    dict_all_ccnorm2_per = data_ddd['dict_all_ccnorm2_per']
    cc_max_all_neurons = data_ddd['cc_max_all_neurons']
    # In[8]:
    print(total_score_sum)

    # In[9]:
    neurons_to_check_global = np.argsort(total_score_sum)[::-1][:5]
    print(neurons_to_check_global)
    global_vars['neurons_to_check_global'] = neurons_to_check_global
    global_vars['cc_max_all_neurons'] = cc_max_all_neurons
    neurons_to_check_this_one_dict = dict()
    show_diff_every_key(dict_all_ccnorm2, dict_all_ccnorm2_per, neurons_to_check_this_one_dict, global_vars)
    global_vars['neurons_to_check_this_one_dict'] = neurons_to_check_this_one_dict
    global_vars['candidates_to_visualize'] = get_candidates_to_visualize()
    visualize_all_cases(
        tv_weight_str=tv_weight_str,
        lp_norm_weight_str=lp_norm_weight_str,
        seed=int(seed_str),
        global_vars=global_vars,
    )


if __name__ == '__main__':
    main()
