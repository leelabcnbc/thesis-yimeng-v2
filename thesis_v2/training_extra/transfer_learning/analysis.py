"""functions to handle data collected"""
from os.path import join, exists
from os import listdir
from json import load, dump
import numpy as np

from ... import dir_dict


def collect_one_setting(
        *,
        sparse_list,
        keygen,
        param,
):

    key_ref = keygen(**param, sparse=sparse_list[0])
    entries = listdir(join(dir_dict['models'], key_ref))

    entries = set(entries)
    entries -= {'results.json', }
    num_neuron = len(entries)
    assert entries == set(f'n{i}' for i in range(num_neuron))

    # then create two arrays of size n_neuron x n_lambda.
    # one for val corr.
    # one for test corr.

    num_sparse = len(sparse_list)
    performance_val = np.full((num_neuron, num_sparse), fill_value=np.nan,
                              dtype=np.float64)
    performance_test = np.full((num_neuron, num_sparse), fill_value=np.nan,
                               dtype=np.float64)

    # then for each sparse and each neuron, fill in the matrices above.
    for sparse_idx, sparse_this in enumerate(sparse_list):
        key_this_prefix = keygen(**param, sparse=sparse_this)
        summary_file = join(dir_dict['models'], key_this_prefix,
                            'results.json')
        if not exists(summary_file):
            val_this = np.full((num_neuron,),
                               fill_value=np.nan, dtype=np.float64
                               )
            test_this = np.full((num_neuron,),
                                fill_value=np.nan, dtype=np.float64
                                )

            for neuron_idx in range(num_neuron):
                key_this = f'{key_this_prefix}/n{neuron_idx}'
                with open(
                        join(dir_dict['models'], key_this, 'stats_best.json'),
                        'r', encoding='utf-8') as f_stats:
                    stats = load(f_stats)
                val_this[neuron_idx] = stats['stats']['val']['corr_mean']
                test_this[neuron_idx] = stats['stats']['test']['corr_mean']
            #             print(sparse_idx, neuron_idx)
            np.all(np.isfinite(val_this)) and np.all(np.isfinite(test_this))
            with open(summary_file, 'w', encoding='utf-8') as f_summary:
                dump(
                    {
                        'val': val_this.tolist(),
                        'test': test_this.tolist(),
                    },
                    f_summary
                )
        with open(summary_file, 'r', encoding='utf-8') as f_summary:
            summary = load(f_summary)
        val_this = np.array(summary['val'])
        test_this = np.array(summary['test'])
        assert val_this.shape == test_this.shape == (num_neuron,)
        performance_val[:, sparse_idx] = val_this
        performance_test[:, sparse_idx] = test_this

    assert np.all(np.isfinite(performance_val))
    assert np.all(np.isfinite(performance_test))

    # use validation set to pick best result
    argmax_this = np.argmax(performance_val, axis=1)

    # then get argmax of performance_val
    test_best = performance_test[np.arange(num_neuron), argmax_this]
    assert test_best.shape == (num_neuron,)

    return {
        'test_best': test_best,
        'argmax': argmax_this,
    }
