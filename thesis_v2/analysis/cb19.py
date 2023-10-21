from strflab.stats import cc_max
from thesis_v2.data.prepared import cadena_plos_cb19

def load_ccmax():
    # get cc_max
    cc_max_all_neurons = cc_max(cadena_plos_cb19.get_neural_data_per_trial(fill_value='zero'))
    assert cc_max_all_neurons.shape == (115,)
    return cc_max_all_neurons
