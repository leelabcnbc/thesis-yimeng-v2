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
