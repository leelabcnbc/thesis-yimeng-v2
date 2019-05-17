"""
most of this file is adapted from
<https://github.com/leelabcnbc/thesis-proposal-yimeng-201808/blob/master/thesis_proposal/data_aux/neural_datasets.py>  # noqa: E501
"""
import os
from scipy.io import loadmat
import numpy as np
from . import dir_dict, join, abspath, register_data

_data_root = abspath(join(dir_dict['private_data'], 'crcns_pvc-8', 'data'))


def ld_neural(section: int):
    # follow `crcns_pvc-8_data_description.pdf` on
    # <http://crcns.org/data-sets/vc/pvc-8>.
    # this script only handles neural data, not images.
    assert section in range(1, 11)
    mat_to_process = loadmat(join(_data_root,
                                  '{:02d}.mat'.format(section)))
    resp_train = mat_to_process['resp_train']
    resp_train_blk = mat_to_process['resp_train_blk']
    centered_neuron_idx = mat_to_process['INDCENT']

    # shift response according to the official demo script.
    latency = 50
    assert latency > 0
    num_neuron, num_im, num_trial, num_time = resp_train.shape
    #  # doc says 106, I found both 105 and 106 for num_time
    assert num_im == 956 and num_trial == 20 and num_time in {105,
                                                              106}

    resp_train_raw = resp_train.copy()

    assert resp_train_blk.ndim == 4 and resp_train_blk.shape == (
        num_neuron, num_im, num_trial, 211)
    assert resp_train_blk.shape[3] > latency
    # then let's shift.
    resp_train = np.concatenate(
        [resp_train[:, :, :, latency:], resp_train_blk[:, :, :, :latency]],
        axis=3)
    assert np.all(np.isfinite(resp_train))
    assert resp_train.shape == (num_neuron, num_im, num_trial, num_time)
    # then sum all responses
    # int64 all the way.
    resp_train = resp_train.sum(axis=3, dtype=np.int64)
    # use my convention. num im x num trial x num neuron.
    resp_train = np.transpose(resp_train, (1, 2, 0))
    assert resp_train.shape == (num_im, num_trial, num_neuron)
    assert centered_neuron_idx.shape == (num_neuron, 1) and set(
        np.unique(centered_neuron_idx)) <= {0, 1}
    centered_neuron_idx = centered_neuron_idx.ravel().astype(np.bool_)

    # add some other properties
    homo = mat_to_process['P_HOMOG']
    assert homo.shape == (num_neuron, 956) and homo.dtype == np.float64
    rf_spatial = mat_to_process['RF_SPATIAL']
    assert rf_spatial.shape == (
        num_neuron, 5) and rf_spatial.dtype == np.float64
    rf_tuning = mat_to_process['RF_TUNING']
    assert rf_tuning.shape == (num_neuron,
                               4) and rf_tuning.dtype == np.float64

    # more type assertion.
    # this is now num_im x num_neuron
    resp_train = resp_train.sum(axis=1, dtype=np.int64)
    assert resp_train.shape == (num_im, num_neuron)
    assert resp_train.dtype == np.int64
    assert resp_train_raw.dtype == np.uint8

    return {
        'mean': resp_train,  # this is shifted, for everyday use,
                             # with 50 latency (actually is sum)
        'all': resp_train_raw,  # this is unshifted.
        'attrs': {
            'INDCENT': centered_neuron_idx,
            'P_HOMOG': homo,
            'RF_SPATIAL': rf_spatial,
            'RF_TUNING': rf_tuning,
        }
    }


# https://stackoverflow.com/questions/10452770/python-lambdas-binding-to-local-values  # noqa: E501
register_data(
    'crcns_pvc-8_neural',
    {'{:02d}'.format(x): lambda x=x: ld_neural(x) for x in range(1, 11)}
)


def ld_neural_blk(section: int):
    # follow `crcns_pvc-8_data_description.pdf` on
    # <http://crcns.org/data-sets/vc/pvc-8>.
    # this script only handles neural data, not images.
    assert section in range(1, 11)
    mat_to_process = loadmat(join(_data_root,
                                  '{:02d}.mat'.format(section)))
    resp_train_blk = mat_to_process['resp_train_blk']

    num_neuron, num_im, num_trial, num_time = resp_train_blk.shape
    # papers says 200, I found 211 time bins
    assert num_im == 956 and num_trial == 20 and num_time == 211
    assert resp_train_blk.dtype == np.uint8
    return {'all': resp_train_blk}


register_data(
    'crcns_pvc-8_neural_blk',
    {'{:02d}'.format(x): lambda x=x: ld_neural_blk(x) for x in range(1, 11)}
)


# check <https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/results_ipynb/gsm_2015_paper_reproduce/images.ipynb> for details  # noqa: E501
def save_img(window_size):
    # large for 6.7, medium for 3
    # check <https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/scripts/debug/crcns_pvc-8/check_images.m>  # noqa: E501
    assert window_size in {'large', 'medium'}
    if window_size == 'large':
        file_to_use = '01.mat'
    elif window_size == 'medium':
        file_to_use = '08.mat'
    else:
        raise ValueError
    data = loadmat(os.path.join(_data_root, file_to_use))[
        'images'].ravel()
    assert data.shape == (956,)
    x_all = np.array([x for x in data])
    assert x_all.shape == (956, 320, 320) and x_all.dtype == np.uint8
    return x_all


register_data('crcns_pvc-8_images', {
    'large': lambda: save_img('large'),
    'medium': lambda: save_img('medium'),
})
