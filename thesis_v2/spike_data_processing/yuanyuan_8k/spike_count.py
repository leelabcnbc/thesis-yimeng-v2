"""mimic the inner loop of get_spike_counts.m"""
import numpy as np


def cdttable_to_spike_count(*,
                            cdttable, neurons_to_extract: list,
                            start_time, end_time,
                            num_condition_debug,
                            ):
    # cdttable is a numpy array with at least field names
    # {
    #     'condition', 'spikeElectrode', 'spikeUnit', 'spikeTimes'
    # }
    # it can done obtained by loadmat('path/to/cdt.mat')['CDTTables'][0,0][0,0]
    #
    #
    #
    # neurons_to_extract is a list of 2-tuples.
    # (electrode, unit)
    #
    #
    # start_time, end_time are time in terms of bins.
    # for 8k data, the setting should be (400, 1500).
    #
    # that would be 1500-400=1100 bins.
    #
    n_neuron = len(neurons_to_extract)

    num_condition = cdttable['condition'].size
    assert num_condition_debug == num_condition
    assert cdttable['condition'].shape == (num_condition, 1)
    # one trial for each condition. this is kind of assumed in yuanyuan's code.
    assert np.array_equal(np.arange(1, num_condition + 1), np.unique(cdttable['condition']))
    assert start_time < end_time
    assert type(start_time) is int and type(end_time) is int
    bins = np.arange(start_time, end_time+1)
    spike_counts = np.zeros((n_neuron, num_condition, end_time - start_time), dtype=np.uint8)

    for trial_idx in range(num_condition):
        condition = cdttable['condition'][trial_idx, 0]
        assert 1 <= condition <= num_condition
        spike_electrode_this = cdttable['spikeElectrode'][trial_idx, 0].ravel()
        spike_unit_this = cdttable['spikeUnit'][trial_idx, 0].ravel()

        assert spike_electrode_this.shape == spike_unit_this.shape == (spike_unit_this.size,)

        for spike_idx, (elec_this, unit_this) in enumerate(zip(spike_electrode_this, spike_unit_this)):
            try:
                find_idx = neurons_to_extract.index((elec_this, unit_this))
            except ValueError:
                # ignore this
                continue
            # get all times.
            times = cdttable['spikeTimes'][trial_idx, 0][spike_idx, 0].ravel()
            data_to_insert = np.histogram(times * 1000, bins=bins)[0]
            assert data_to_insert.shape == (end_time-start_time,)
            assert np.all(data_to_insert >= 0)
            assert np.all(data_to_insert <= 255)
            spike_counts[find_idx, condition-1, :] = data_to_insert

    return spike_counts
