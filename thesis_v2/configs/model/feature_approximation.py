from os.path import join

from ...submission import utils
from .maskcnn_polished_with_local_pcn import (
    explored_models_summer_2019_certain,
    keygen as keygen_s,
    script_keygen as script_keygen_s,
)
from .maskcnn_polished_with_rcnn_k_bl import (
    explored_models_20200218,
    keygen as keygen_k,
    script_keygen as script_keygen_k,
)

from ... import dir_dict

consts = {
    'local_pcn_recurrent_sep2_model_prefix': 'feature_approximation_lpcn_recurrent_sep2',
    # for configs in scripts/feature_extraction/yuanyuan_8k_a/maskcnn_polished_with_local_pcn/certain_configs.py
    'local_pcn_feature_extraction_certain_configs_yuanyuan_8k_a': {
        'feature_file_dir': join(dir_dict['features'],
                                 'maskcnn_polished_with_local_pcn',
                                 'certain_configs'),
        'augment_config': {
            'module_names': ['bottomup', 'topdown', 'final'],
            'name_mapping': {
                'moduledict.conv1.lambda_out': 'bottomup',
                'moduledict.conv1.lambda_in': 'topdown',
                'moduledict.final_act': 'final',
            }
        }
    },
    'local_pcn_original_imagenet_imagenet_val': {
        'feature_file': join(dir_dict['features'], 'cnn_feature_extraction', 'imagenet_val', 'pcn_local.hdf5'),
        'dataset_prefix_prefix': 'imagenet_val',
        'model_name': 'PredNetBpE_3CLS',
    },
    'local_pcn_original_imagenet_sep2_model_prefix': 'PredNetBpE_3CLS_sep2',
    'k_bl_feature_extraction_20200218_yuanyuan_8k_a': {
        # consistent with
        # scripts/feature_extraction/yuanyuan_8k_a/maskcnn_polished_with_rcnn_k_bl/20200218.py
        'feature_file_dir': join(
            # for cnbc cluster, whose `/user_data/yimengzh` is not big enough.
            # '/home/yimengzh/thesis-v2-large-files',
            dir_dict['features'],
            'maskcnn_polished_with_rcnn_k_bl',
            '20200218'),
        'augment_config': {
            'module_names': ['layer0', 'layer1', 'layer2'],
            'name_mapping': {
                'moduledict.bl_stack.input_capture': 'layer0',
                'moduledict.bl_stack.capture_list.0': 'layer1',
                'moduledict.bl_stack.capture_list.1': 'layer2',
            }
        }
    },
    'k_bl_recurrent_k3_sep2+0_model_prefix': 'feature_approximation_kbl_recurrent_k3_sep2+0',
    'k_bl_recurrent_k3_sep2+1_model_prefix': 'feature_approximation_kbl_recurrent_k3_sep2+1',
}


def local_pcn_recurrent_sep2_hyparameters():
    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'model_seed',
        range(1),
    )

    param_iterator_obj.add_pair(
        'act_fn',
        # should try relu later
        ('relu', 'softplus'),
    )

    param_iterator_obj.add_pair(
        'loss_type',
        ('mse', 'l1')  # should try mse later
    )

    # In theory, it should be 9. but let's also try smaller ones.
    param_iterator_obj.add_pair(
        'kernel_size',
        (7, 9,)
    )

    # pcn_bypass={pcn_bypass},
    param_iterator_obj.add_pair(
        'bn_pre',
        (True,),
    )

    return param_iterator_obj


def k_bl_recurrent_sep2_hyparameters():
    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'model_seed',
        range(1),
    )

    param_iterator_obj.add_pair(
        'act_fn',
        # should try relu later
        ('relu', 'softplus'),
    )

    param_iterator_obj.add_pair(
        'loss_type',
        ('mse', 'l1')  # should try mse later
    )

    # In theory, it should be 9. but let's also try smaller ones.
    param_iterator_obj.add_pair(
        'kernel_size',
        (9,)
    )

    # pcn_bypass={pcn_bypass},
    param_iterator_obj.add_pair(
        'bn_pre',
        (True,),
    )

    param_iterator_obj.add_pair(
        'bn_before_act',
        (False, True),
    )

    return param_iterator_obj


def local_pcn_original_imagenet_sep2_hyparameters():
    param_iterator_obj = utils.ParamIterator()

    param_iterator_obj.add_pair(
        'model_seed',
        range(1),
    )

    param_iterator_obj.add_pair(
        'act_fn',
        # should try relu later
        ('relu', 'softplus'),
    )

    param_iterator_obj.add_pair(
        'loss_type',
        ('mse', 'l1')  # should try mse later
    )

    # In theory, it should be 9. but let's also try smaller ones.
    param_iterator_obj.add_pair(
        'kernel_size',
        (9,)
    )

    # pcn_bypass={pcn_bypass},
    param_iterator_obj.add_pair(
        'bn_pre',
        (True,),
    )

    return param_iterator_obj


def model_params_local_pcn_recurrent_summer_2019_certain(good_model_param) -> dict:
    all_params_dict = dict()
    for idx, param in enumerate(explored_models_summer_2019_certain().generate()):
        # let's use a fully recurrent one for debugging.
        if not good_model_param(param):
            continue

        key = keygen_s(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})
        key_script = script_keygen_s(**param)
        all_params_dict[key_script] = {
            'key': key,
            'param': param,
            'idx': idx,  # this idx will be used to identify models. otherwise names are too long.
        }

    return all_params_dict


def model_params_b_kl_recurrent_20200218(good_model_param) -> dict:
    all_params_dict = dict()
    for idx, param in enumerate(explored_models_20200218().generate()):
        # let's use a fully recurrent one for debugging.
        if not good_model_param(param):
            continue

        key = keygen_k(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})
        key_script = script_keygen_k(**param)
        all_params_dict[key_script] = {
            'key': key,
            'param': param,
            'idx': idx,  # this idx will be used to identify models. otherwise names are too long.
        }

    return all_params_dict


def keygen(*,
           model_seed: int,
           act_fn: str,
           loss_type: str,
           kernel_size: int,
           bn_pre: bool,
           basemodel_idx: int,
           dataset_prefix: str = 'yuanyuan_8k_a_3day',
           model_prefix: str,
           bn_before_act: bool = False,
           ):
    if not bn_before_act:
        # for modeling residuals
        return f'{dataset_prefix}/{model_prefix}/baseidx{basemodel_idx}/act{act_fn}/loss{loss_type}/k{kernel_size}/bn_pre{bn_pre}/model_seed{model_seed}'  # noqa: E501
    else:
        # for modeling non negative outputs
        return f'{dataset_prefix}/{model_prefix}/baseidx{basemodel_idx}/act{act_fn}/loss{loss_type}/k{kernel_size}/bn_pre{bn_pre}/bn_b4_act{bn_before_act}/model_seed{model_seed}'  # noqa: E501


def script_keygen(*,
                  return_original_key=False,
                  return_model_prefix=False,
                  **kwargs):
    del kwargs['basemodel_key_script']
    key = keygen(**kwargs)

    # remove dataset_prefix/model_prefix part
    if not return_model_prefix:
        ret = '+'.join(key.split('/')[2:])
    else:
        ret = '+'.join(key.split('/')[1:])
    if return_original_key:
        return ret, key
    else:
        return ret
