from os.path import join

from ...submission import utils
from .maskcnn_polished_with_local_pcn import (
    explored_models_summer_2019_certain,
    keygen as keygen_s,
    script_keygen as script_keygen_s,
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
    }
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


def keygen(*,
           model_seed: int,
           act_fn: str,
           loss_type: str,
           kernel_size: int,
           bn_pre: bool,
           basemodel_idx: int,
           dataset_prefix: str = 'yuanyuan_8k_a_3day',
           model_prefix: str,
           ):
    return f'{dataset_prefix}/{model_prefix}/baseidx{basemodel_idx}/act{act_fn}/loss{loss_type}/k{kernel_size}/bn_pre{bn_pre}/model_seed{model_seed}'  # noqa: E501


def script_keygen(*, return_original_key=False, **kwargs):
    del kwargs['basemodel_key_script']
    key = keygen(**kwargs)

    # remove dataset_prefix/model_prefix part
    ret = '+'.join(key.split('/')[2:])
    if return_original_key:
        return ret, key
    else:
        return ret
