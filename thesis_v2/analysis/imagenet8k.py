import numpy as np
from strflab.stats import cc_max
from thesis_v2.data.prepared.yuanyuan_8k import get_neural_data_per_trial

def load_ccmax():
    # get cc_max
    cc_max_all_neurons = cc_max(get_neural_data_per_trial(('042318', '043018', '051018',)))
    assert cc_max_all_neurons.shape == (79,)
    return cc_max_all_neurons

def load_ccmax_v1():
    # get cc_max
    data = get_neural_data_per_trial(('042318', '043018', '051018',))
    v1_ind = np.array([8, 9, 10, 11, 12, 13, 14, 18, 34, 35, 36, 37, 38, 39, 40, 
                       45, 54, 60, 61, 62, 63, 64, 65, 66, 68])
    data = data[v1_ind]
    cc_max_all_neurons = cc_max(data)
    assert cc_max_all_neurons.shape == (25,)
    return cc_max_all_neurons

def load_ccmax_v2():
    # get cc_max
    data = get_neural_data_per_trial(('042318', '043018', '051018',))
    v1_ind = np.array([8, 9, 10, 11, 12, 13, 14, 18, 34, 35, 36, 37, 38, 39, 40, 
                       45, 54, 60, 61, 62, 63, 64, 65, 66, 68])
    v2_ind = list(set(range(79)) - set(v1_ind))
    data = data[v2_ind]
    cc_max_all_neurons = cc_max(data)
    assert cc_max_all_neurons.shape == (54,)
    return cc_max_all_neurons