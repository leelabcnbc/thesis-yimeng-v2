from sys import path
from os.path import join, dirname, exists
from os import makedirs
from thesis_v2 import dir_dict

folder_to_check = 'scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn'
path.insert(0, join(dir_dict['root'], folder_to_check))
from thesis_v2.models.maskcnn_polished_with_local_pcn.builder import load_modules
from submit_certain_configs import param_iterator_obj
from key_utils import keygen, script_keygen

load_modules()

from thesis_v2.training.training_aux import load_training_results
from thesis_v2.data.prepared.yuanyuan_8k import get_data
from thesis_v2.training_extra.data import generate_datasets

from thesis_v2.training_extra.evaluation import eval_fn_wrapper as eval_fn_wrapper_neural

from thesis_v2.training.training import eval_wrapper
from functools import partial
from torchnetjson.builder import build_net
import numpy as np
from numpy.random import RandomState
from torch.backends import cudnn
import torch
assert torch.__version__ in {'1.3.1'}

cudnn.enabled = True
cudnn.deterministic = True
cudnn.benchmark = False


def gen_data(size):
    rs = RandomState(seed=0)
    return torch.tensor(rs.uniform(0, 1, (1, 1, size, size)).astype(np.float32)).cuda()


def do_one(param):
    # this is tricky, because I did not save per neuron corr during training.
    # shit.
    #
    # afterwards, I will simply save thoese perneuron corr.
    #
    # right now, let's load the data set and do the evaluation again.
    assert param['out_channel'] == 16
    assert param['num_layer'] == 2
    assert param['kernel_size_l1'] == 9
    assert param['pooling_ksize'] == 3
    assert param['pooling_type'] == 'avg'

    # assert param['model_seed'] == 0

    key = keygen(**{k: v for k, v in param.items() if k not in {'scale', 'smoothness'}})

    # then output model to onnx, write to the same location
    model_loc = join(dir_dict['models'], 'onnx_export', key, 'model.onnx')
    print(model_loc)
    makedirs(dirname(model_loc), exist_ok=True)
    if exists(model_loc):
        return

    result = load_training_results(key, return_model=False)
    # load twice, first time to get the model.
    result = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))

    # then create data set.
    datasets = get_data('a', 200, param['input_size'], ('042318', '043018', '051018'), scale=0.5,
                        seed=param['split_seed'])
    datasets = {
        'X_train': datasets[0].astype(np.float32),
        'y_train': datasets[1],
        'X_val': datasets[2].astype(np.float32),
        'y_val': datasets[3],
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }

    # only the test one is needed.
    datasets = generate_datasets(
        **datasets,
        per_epoch_train=True, shuffle_train=True,
    )['test']

    result_on_the_go = eval_wrapper(result['model'].cuda(),
                                    datasets,
                                    'cuda',
                                    1,
                                    partial(eval_fn_wrapper_neural, loss_type=param['loss_type']),
                                    lambda dummy1, dummy2, dummy3: torch.tensor(0.0)
                                    )

    # check metrics match on PyTorch 1.3.1 side; the model was trained on PyTorch 1.1.0
    assert abs(result_on_the_go['corr_mean'] - result['stats_best']['stats']['test']['corr_mean']) < 1e-4

    # then export
    input_dummy = gen_data(param['input_size'])
    # set weight
    result['model'].moduledict['fc'].set_weight_for_eval()
    torch.onnx.export(result['model'], [input_dummy], model_loc,
                      verbose=True, input_names=['input'],
                      output_names=['output'],
                      # for https://github.com/pytorch/pytorch/issues/30289
                      keep_initializers_as_inputs=True,
                      # this part should really be added, but for our purpose,
                      # it's not necessary, since onnx2keras does not care about it.
                      # dynamic_axes={'input': [0, ], 'output': [0, ]}
                      )


def main():
    model_count = len(list(param_iterator_obj.generate()))
    # this should be 2304, which is 128 (configs) * 3 (seeds) * 6 (number of cycles from 0-5)
    assert model_count == 2304
    for idx, param_dict in enumerate(param_iterator_obj.generate()):
        print(f'model {idx}/{model_count}')
        do_one(param_dict)


if __name__ == '__main__':
    main()
