# follow https://github.com/leelabcnbc/thesis-yimeng-v1/blob/0eafb103e1a92b4dc706c6f245ef1dca81351845/scripts/debug/transfer_learning/transfer_learning.ipynb  # noqa: E501

from tempfile import TemporaryDirectory
from os import environ
# import gc

import h5py
import numpy as np

from joblib import Parallel, delayed
from thesis_v2 import dir_dict, join

from thesis_v2.data.prepared import split_stuff
from thesis_v2.data.prepared.yuanyuan_8k import (
    get_neural_data, get_indices
)

from thesis_v2.models.transfer_learning.builder import (gen_transfer_learner,
                                                        load_modules)

from thesis_v2.training_extra.transfer_learning.training import (train_one,
                                                                 partial)

from thesis_v2.training_extra.transfer_learning.opt import (
    get_transfer_learning_opt_config
)

from thesis_v2.training.utils import get_memmapped_array

from key_utils import keygen

file_to_save_feature = join(
    dir_dict['features'],
    'cnn_feature_extraction',
    'yuanyuan_8k_a',
    'vgg.hdf5',
)

load_modules()


def handle_one_case(x_train, y_train, x_val, y_val, x_test, y_test, act_fn,
                    model_seed, key, sparse, neuron_idx, loss_type):
    dataset_this = {
        'X_train': x_train,
        'y_train': y_train[:, neuron_idx:neuron_idx + 1],
        'X_val': x_val,
        'y_val': y_val[:, neuron_idx:neuron_idx + 1],
        'X_test': x_test,
        'y_test': y_test[:, neuron_idx:neuron_idx + 1],
    }

    load_modules()

    def gen_cnn_partial(in_shape, n):
        return gen_transfer_learner(in_shape, n,
                                    batchnorm=True,
                                    batchnorm_affine=False,
                                    act_fn=act_fn,
                                    factorized=True
                                    )

    opt_config_partial = partial(get_transfer_learning_opt_config,
                                 sparse=float(sparse), loss_type=loss_type)
    #
    res = train_one(arch_json_partial=gen_cnn_partial,
                    opt_config_partial=opt_config_partial,
                    datasets=dataset_this,
                    # note this gets saved under v1 folder...
                    # but it should not matter.
                    key=f'{key}/n{neuron_idx}',
                    show_every=100000000,  # no output.
                    model_seed=model_seed,
                    return_model=False)

    return res['stats_best']['stats']['test']['corr_mean']


def master(*,
           split_seed,
           sparse,
           model_seed,
           act_fn,
           loss_type,
           suffix,
           ):

    key = keygen(
        split_seed=split_seed,
        sparse=sparse,
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        suffix=suffix,
    )

    neural_all = get_neural_data(('042318', '043018', '051018'),
                                 scale=0.5)
    num_neuron = neural_all.shape[1]
    assert neural_all.dtype == np.float64
    idx_stuff = get_indices('a', split_seed)
    assert len(idx_stuff) == 3
    print(
        'train', idx_stuff[0].size,
        'val', idx_stuff[1].size,
        'test', idx_stuff[2].size,
        'num_neuron', num_neuron,
    )

    with TemporaryDirectory(dir=environ.get('JOBLIB_TEMP_FOLDER', None)) as d:
        print(f'array saved in {d}')

        with h5py.File(file_to_save_feature, 'r') as f:
            x_all = f['/'.join(['a', suffix])][...]

        if x_all.ndim != 4:
            assert x_all.ndim == 2
            x_all = x_all[..., np.newaxis, np.newaxis]
        assert x_all.ndim == 4
        print(x_all.shape, neural_all.shape)

        dataset_all = split_stuff(x_all, neural_all, idx_stuff)
        assert len(dataset_all) == 6

        try:
            # toggle this on/off to see how much memory I can save.
            # (with smem https://www.selenic.com/smem/)
            dataset_all = get_memmapped_array(dataset_all, d)

            # 6 jobs matches slurm submission file in terms of # of CPUs.
            results = Parallel(
                # 6 was used in most cases (all `quarter` and most `half`).
                # n_jobs=6,
                # MODIFY this to 4 for some large `half` cases (lower layers)
                # fuck CNBC cluster for the memory calculation issue.
                n_jobs=4,
                verbose=3,
                # if you use this, then memory from x_all in the main process
                # will count as in shared memory as well.
                # lorky does not have this problem, I believe lorky does not
                # use fork, and in doing so some imports are ignored
                # backend='multiprocessing'
            )(delayed(handle_one_case)(
                dataset_all[0], dataset_all[1],
                dataset_all[2], dataset_all[3],
                dataset_all[4], dataset_all[5],
                act_fn, model_seed, key, sparse, idx, loss_type,
            ) for idx in range(num_neuron))

            print(np.array(results).mean())
        finally:
            # otherwise, memmap file is not closed (`del` is the correct way
            # to close it, according to numpy doc.

            # if not closed, then TemporaryDirectory's cleanup, which
            # uses shutil.rmtree, can fail
            # see https://bugs.launchpad.net/os-win/+bug/1669335
            del dataset_all
