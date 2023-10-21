from strflab.stats import cc_max
from thesis_v2.data.prepared.gaya import get_neural_data
import numpy as np

def load_ccmax():
    # get cc_max
    raw_data = get_neural_data(dataset='tang', return_raw=True, start_offset=0, end_offset=500)
    # use first eight trials to get ccmax.
    # each image at least has 8 trials. this will be a bit inaccurate but should be ok. with similar results as
    # a more proper way to get ccmax.
    a = np.asarray([x[:8] for x in raw_data]).T
    cc_max_all_neurons = cc_max(a)
    assert cc_max_all_neurons.shape == (34,)
    return cc_max_all_neurons

def load_ccmax_v1():
    # get cc_max
    raw_data = get_neural_data(dataset='tang', return_raw=True, start_offset=0, end_offset=500)
    # use first eight trials to get ccmax.
    # each image at least has 8 trials. this will be a bit inaccurate but should be ok. with similar results as
    # a more proper way to get ccmax.
    a = np.asarray([x[:8] for x in raw_data]).T
    v1_ind = np.array([2, 6, 9, 22, 25, 31])
    a = a[v1_ind, :, :]
    cc_max_all_neurons = cc_max(a)
    assert cc_max_all_neurons.shape == (6,)
    return cc_max_all_neurons

def load_ccmax_v2():
    # get cc_max
    raw_data = get_neural_data(dataset='tang', return_raw=True, start_offset=0, end_offset=500)
    a = np.asarray([x[:8] for x in raw_data]).T
    v1_ind = np.array([2, 6, 9, 22, 25, 31])
    v2_ind = list(set(range(34)) - set(v1_ind))
    a = a[v2_ind, :, :]
    cc_max_all_neurons = cc_max(a)
    assert cc_max_all_neurons.shape == (28,)
    return cc_max_all_neurons