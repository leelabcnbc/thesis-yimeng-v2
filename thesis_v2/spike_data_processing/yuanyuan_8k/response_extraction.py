"""extract responses from 500 movies to 8000 images"""

import numpy as np


def extract_response(
        *,
        spike_count,
        sort_index,
        time_delay,
        num_frame,
        extraction_length,
        frame_time: float,
        neuron_list=None,
        normaliztion_config=None,
):
    """

    :param spike_count: 3d np.ndarray (n_neuron, n_movie, n_timebin)
    :param sort_index: a permutation of range(num_frame*n_movie)
    :param time_delay: offset for extraction
    :param num_frame:
    :param extraction_length: how long to extract for each frame
    :param frame_time: how long is each frame presented.
    :param neuron_list: which set of neurons to extract. if None, it's range(n_neuron)
    :param normaliztion_config: how to normalize response.
    :return: (len(neuron_list), n_movie*num_frame, extraction_length) array.


    the order of neurons is based on neuron_list.
    """
    assert type(time_delay) is int
    assert type(num_frame) is int
    assert type(extraction_length) is int
    assert type(frame_time) is float

    n_neuron, n_movie, n_timebin = spike_count.shape

    if neuron_list is None:
        neuron_list = np.arange(n_neuron)

    result = np.zeros((len(neuron_list), n_movie, num_frame),
                      dtype=np.float64)

    assert sort_index.shape == (n_movie * num_frame,)
    assert np.array_equal(np.arange(n_movie * num_frame), np.sort(sort_index))

    for idx_frame in range(num_frame):
        crop_start = round(time_delay + idx_frame * frame_time)
        crop_end = crop_start + extraction_length
        assert 0 <= crop_start < crop_end <= n_timebin
        result[:, :, idx_frame] = spike_count[neuron_list, :, crop_start:crop_end].mean(axis=-1)

    # normalization

    if normaliztion_config == 'legacy':
        # use yuanyuan's way.
        # 0.00001 looks like magic. I think that might be used to deal with zero response.
        # best way is to just replace 0 with 1 in the divisor.
        result /= result.mean(axis=1, keepdims=True) + 0.00001
    elif normaliztion_config is not None:
        raise NotImplementedError

    # then reshape.
    result = result.reshape((len(neuron_list), n_movie * num_frame))
    result = result[:, sort_index]

    assert result.shape == (len(neuron_list), n_movie * num_frame)

    return result
