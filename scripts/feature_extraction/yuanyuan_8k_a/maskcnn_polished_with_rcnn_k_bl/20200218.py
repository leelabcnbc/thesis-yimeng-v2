from os.path import join, dirname
from os import makedirs
# from collections import OrderedDict

import numpy as np
from torch import tensor
import h5py

from torchnetjson.builder import build_net
from thesis_v2.training.training_aux import load_training_results
from thesis_v2 import dir_dict
from thesis_v2.data.prepared.yuanyuan_8k import get_data
from thesis_v2.feature_extraction.extraction import extract_features
from thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder import load_modules

from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    explored_models_20200218,
    script_keygen,
    keygen
)

load_modules()

global_vars = {
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
}


def good_model_param(param):
    return param['rcnn_bl_cls'] == 4 and param['kernel_size_l23'] == 3 and param['num_layer'] == 3


def get_all_model_params():
    all_params_dict = dict()
    for idx, param in enumerate(explored_models_20200218().generate()):
        # let's use a fully recurrent one for debugging.
        if not good_model_param(param):
            continue

        key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})
        key_script = script_keygen(**param)
        all_params_dict[key_script] = {
            'key': key,
            'param': param,
        }

    return all_params_dict


_load_dataset_cache = dict()


def load_dataset_cached(*, input_size, split_seed):
    key = (input_size, split_seed)
    if key not in _load_dataset_cache:
        _load_dataset_cache[key] = get_data('a', 200, input_size, ('042318', '043018', '051018'), scale=0.5,
                                            seed=split_seed)
        _load_dataset_cache[key] = {
            'X_train': _load_dataset_cache[key][0].astype(np.float32),
            # 'X_val': _load_dataset_cache[key][2].astype(np.float32),
            # 'X_test': _load_dataset_cache[key][4].astype(np.float32),
        }

    return _load_dataset_cache[key]


# then process one model by a model
def process_one_model(*, key_script, key, param):
    # load data set
    data = load_dataset_cached(input_size=param['input_size'], split_seed=param['split_seed'])

    # load model
    result = load_training_results(key, return_model=False)
    # load twice, first time to get the model.
    model = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))['model']

    model.cuda()
    model.eval()

    for dataset_name, dataset in data.items():
        print(f'process {key_script}/{dataset_name}')
        process_one_model_one_dataset(
            model=model, dataset_to_extract=dataset, dataset_name=dataset_name,
            key_script=key_script,
        )


def process_one_model_one_dataset(*, model, dataset_to_extract, dataset_name, key_script):
    grp_name = dataset_name
    file_to_save = join(global_vars['feature_file_dir'], key_script + '.hdf5')
    augment_config = global_vars['augment_config']
    makedirs(dirname(file_to_save), exist_ok=True)
    with h5py.File(file_to_save, 'a') as f_feature:
        if grp_name not in f_feature:
            grp = f_feature.create_group(grp_name)

            extract_features(model, (dataset_to_extract,),
                             preprocessor=lambda x: (tensor(x[0]).cuda(),),
                             output_group=grp,
                             batch_size=256,
                             augment_config=augment_config,
                             # mostly for replicating old results
                             deterministic=True,
                             # much faster.
                             flush=True,
                             compression=False,
                             )
        else:
            print('done before!')


def main():
    all_params_dict = get_all_model_params()
    print(f'{len(all_params_dict)} models to process')
    for key_script, value in all_params_dict.items():
        process_one_model(key_script=key_script,
                          key=value['key'],
                          param=value['param'])


if __name__ == '__main__':
    main()
