"""a modification of
https://github.com/leelabcnbc/thesis-yimeng-v1/blob/master/scripts/debug/transfer_learning/transfer_learning_factorized.ipynb

this should be run with thesis-v2 code.
"""  # noqa: E501

from tempfile import TemporaryDirectory
from os import environ
from sys import argv

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

from thesis_v2.training.utils import get_memmapped_array

from thesis_v2.training_extra.transfer_learning.training import (train_one,
                                                                 partial)

from thesis_v2.training_extra.transfer_learning.opt import (
    get_transfer_learning_opt_config
)

file_to_save_feature = join(
    dir_dict['features'],
    'cnn_feature_extraction',
    'yuanyuan_8k_a',
    'vgg.hdf5',
)

load_modules()


def handle_one_case(x_train, y_train, x_val, y_val, x_test, y_test, act_fn,
                    sparse, neuron_idx, note):

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
                                 sparse=float(sparse))
    #
    res = train_one(arch_json_partial=gen_cnn_partial,
                    opt_config_partial=opt_config_partial,
                    datasets=dataset_this,
                    # note this gets saved under v1 folder...
                    # but it should not matter.
                    key=f'debug/transfer_learning/v2_code/{note}/{act_fn}/{sparse}/{neuron_idx}',
                    show_every=100000000,  # no output.
                    model_seed=0, return_model=False)

    return res


def load_dataset_for_cnnpre(act_fn, sparse, suffix, note):
    neural_all = get_neural_data(('042318', '043018', '051018'),
                                 scale=0.5)
    num_neuron = neural_all.shape[1]
    assert num_neuron == 79
    assert neural_all.dtype == np.float64
    idx_stuff = get_indices('a', 0)

    # then load the data in that particular suffix, and then split it according to idx_xxx.
    with h5py.File(file_to_save_feature, 'r') as f:
        x_all = f['/'.join(['a', suffix])][...]
    print(x_all.shape, neural_all.shape)
    assert x_all.ndim == 4
    dataset_all = split_stuff(x_all, neural_all, idx_stuff)

    # then for loop, to get result for each neuron.
    with TemporaryDirectory(dir=environ.get('JOBLIB_TEMP_FOLDER', None)) as d:
        try:
            # toggle this on/off to see how much memory I can save.
            # (with smem https://www.selenic.com/smem/)
            dataset_all = get_memmapped_array(dataset_all, d)

            # 6 jobs matches slurm submission file in terms of # of CPUs.
            results = Parallel(
                n_jobs=6,
                verbose=3,
            )(
                delayed(handle_one_case)(
                    dataset_all[0], dataset_all[1],
                    dataset_all[2], dataset_all[3],
                    dataset_all[4], dataset_all[5],
                    act_fn, sparse, idx,
                    note,
                    # just work on 10 neurons.
                ) for idx in range(10)
            )
        finally:
            # otherwise, memmap file is not closed (`del` is the correct way
            # to close it, according to numpy doc.

            # if not closed, then TemporaryDirectory's cleanup, which
            # uses shutil.rmtree, can fail
            # see https://bugs.launchpad.net/os-win/+bug/1669335
            del dataset_all

    print([r['stats_best']['stats']['test']['corr_mean'] for r in results])


if __name__ == '__main__':
    # randomly pick a small layer.
    assert len(argv) == 2
    # argv[1] can be used to denote which image
    # say, `image-v1` or `image-v2`
    load_dataset_for_cnnpre('softplus', 0.01, 'vgg16/quarter/9.0', argv[1])

# on 2080ti machine,
# for image-v1,
# (leelabcnbc_yimeng-thesis_cu100-2019-02-07-8c53c104a86e.simg)
# I get [0.11745454370975494, 0.3613729774951935, 0.4619869887828827,
#        0.0, 0.0, 0.21956443786621094,
#        0.20594245195388794, 0.4076019525527954, 0.43211671710014343,
#        0.5392784476280212]

# for image-v2
# (yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg)
# I get [0.11718642711639404, 0.3613729774951935, 0.46211448311805725,
#        0.0, 0.0, 0.22203853726387024,
#        0.20600625872612, 0.4079694151878357, 0.43211671710014343,
#        0.5410352945327759]
