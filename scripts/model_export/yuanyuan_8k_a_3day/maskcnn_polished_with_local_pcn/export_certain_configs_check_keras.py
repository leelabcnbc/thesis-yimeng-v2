from sys import path
from os.path import join, exists
from thesis_v2 import dir_dict
import numpy as np
from scipy.stats import pearsonr
from json import dump

from sys import argv

folder_to_check = 'scripts/training/yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn'
path.insert(0, join(dir_dict['root'], folder_to_check))
from submit_certain_configs import param_iterator_obj
from key_utils import keygen, script_keygen

def do_one(param):
    from thesis_v2.training.training_aux import load_training_results
    from thesis_v2.data.prepared.yuanyuan_8k import get_data
    from thesis_v2.training_extra.evaluation import eval_fn_wrapper
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
    result_loc = join(dir_dict['models'], 'onnx_export', key, 'check.json')
    if exists(result_loc):
        return
    print(model_loc)

    import onnx
    from onnx2keras import onnx_to_keras

    result = load_training_results(key, return_model=False)

    # then create data set.
    datasets = get_data('a', 200, param['input_size'], ('042318', '043018', '051018'), scale=0.5,
                        seed=param['split_seed'])
    datasets = {
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }
    model = onnx.load(model_loc)
    k_model = onnx_to_keras(model, ['input'], verbose=False)
    y_test_hat = k_model.predict(datasets['X_test'], batch_size=256)
    result_on_the_go = eval_fn_wrapper(yhat_all=[[y_test_hat]], y_all=[[datasets['y_test']]],
                                       loss_type=param['loss_type'])

    # check metrics match on original code.
    corr_all = pearsonr(np.asarray(result_on_the_go['corr']),
                        np.asarray(result['stats_best']['stats']['test']['corr']))[0]
    print(corr_all, result_on_the_go['corr_mean'], result['stats_best']['stats']['test']['corr_mean'],
          result_on_the_go['corr'][:10], result['stats_best']['stats']['test']['corr'][:10])
    if not np.isnan(corr_all):
        assert corr_all > 0.98
    else:
        # all neurons have the same output. check if their mean values match close enough.
        assert abs(result_on_the_go['corr_mean'] - result['stats_best']['stats']['test']['corr_mean']) < 1e-3
    # print(result_on_the_go['corr_mean'], result['stats_best']['stats']['test']['corr_mean'])
    # assert abs(result_on_the_go['corr_mean'] - result['stats_best']['stats']['test']['corr_mean']) < 1e-3
    data = {
        'keras': {
            'corr': result_on_the_go['corr'],
            'corr_mean': result_on_the_go['corr_mean'],
        },
        'pytorch1.1': {
            'corr': result['stats_best']['stats']['test']['corr'],
            'corr_mean': result['stats_best']['stats']['test']['corr_mean'],
        },
        'corr_between_corr': corr_all,
    }
    with open(result_loc, 'wt', encoding='utf-8') as f_out:
        dump(data, f_out)


def main():
    if len(argv) > 1:
        idx_to_check, = argv[1:]
        idx_to_check = int(idx_to_check)
        assert 0 <= idx_to_check < 2304
    else:
        idx_to_check = None
    model_count = len(list(param_iterator_obj.generate()))
    # this should be 2304, which is 128 (configs) * 3 (seeds) * 6 (number of cycles from 0-5)
    assert model_count == 2304
    for idx, param_dict in enumerate(param_iterator_obj.generate()):
        if idx_to_check is not None and idx != idx_to_check:
            continue
        print(f'model {idx}/{model_count}')
        do_one(param_dict)


if __name__ == '__main__':
    main()
