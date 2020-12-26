from strflab.stats import cc_max
from thesis_v2.data.prepared.yuanyuan_8k import get_neural_data_per_trial

def load_ccmax():
    # get cc_max
    cc_max_all_neurons = cc_max(get_neural_data_per_trial(('042318', '043018', '051018',)))
    assert cc_max_all_neurons.shape == (79,)
    return cc_max_all_neurons
