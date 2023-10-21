from strflab.stats import cc_max
from thesis_v2.data.prepared.tang_v1 import get_neural_trials
import numpy as np

num_neurons = {
    'm1s1': 302,
    'm3s1': 324,
}

def load_ccmax_m3s1():
    # get cc_max
    raw_data = get_neural_trials(site='m3s1')
    cc_max_all_neurons = cc_max(raw_data)
    assert cc_max_all_neurons.shape == (num_neurons['m3s1'],)
    return cc_max_all_neurons