import numpy as np
from modeling.data_utils import get_images, train_val_test_split
from analysis.data_utils import get_all_neural_data, spike_counts, trial_average

# legacy means the set up in
# https://github.com/leelabcnbc/gaya-data/blob/61f21849db0b195d95dda95b224b908206533026/modeling/scripts/train_data_driven_cnn.py   # noqa: E501

global_dict = {
    'legacy_num_img': 5850,
    'legacy_num_img_train': 3800,
    'legacy_num_img_val': 1000,
    'legacy_num_img_test': 1050,
    'legacy_num_neuron': 14,
    'legacy_imsize': 63,
}

assert global_dict['legacy_num_img'] == (
        global_dict['legacy_num_img_train'] + global_dict['legacy_num_img_test'] + global_dict['legacy_num_img_val']
)


def images():
    downsample = 4
    DATASET = 'both'
    x_all = get_images(DATASET, downsample=downsample, torch_format=True,
                       normalize=False)
    assert x_all.shape == (global_dict['legacy_num_img'], 1, global_dict['legacy_imsize'], global_dict['legacy_imsize'])
    assert x_all.min() >= 0
    assert x_all.max() <= 255
    return x_all


def get_neural_data(
        *,
        unit_mean_per_neuron=True,
        post_scale=None,
):
    CORR_THRESHOLD = 0.7
    y = get_all_neural_data(corr_threshold=CORR_THRESHOLD,
                            elecs=False)
    # early response, for all the course project stuff
    y = spike_counts(y, start=540, end=640)
    y = trial_average(y)

    assert y.shape == (
        global_dict['legacy_num_img'],
        global_dict['legacy_num_neuron']
    )

    assert np.all(np.isfinite(y))

    if unit_mean_per_neuron:
        # this is the same type of processing done in 8k data
        per_neuron_mean = y.mean(axis=0)
        assert np.all(per_neuron_mean > 0)
        y = y / per_neuron_mean

    if post_scale is not None:
        y = y * post_scale

    assert y.shape == (
        global_dict['legacy_num_img'],
        global_dict['legacy_num_neuron']
    )

    return y


def get_indices(*, seed):
    assert seed == 'legacy'
    train_idx, val_idx, test_idx = train_val_test_split(
        total_size=5850, train_size=3800, val_size=1000,
        deterministic=True
    )
    assert train_idx.shape == (global_dict['legacy_num_img_train'],)
    assert val_idx.shape == (global_dict['legacy_num_img_val'],)
    assert test_idx.shape == (global_dict['legacy_num_img_test'],)
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx, test_idx])),
                          np.arange(global_dict['legacy_num_img']))

    return train_idx, val_idx, test_idx


def get_data(*, seed, scale=None):
    x_all = images()
    assert x_all.shape == (global_dict['legacy_num_img'], 1, global_dict['legacy_imsize'], global_dict['legacy_imsize'])

    y = get_neural_data(post_scale=scale)

    indices = get_indices(seed=seed)

    result = []
    for idx in indices:
        result.append(x_all[idx])
        if isinstance(y, np.ndarray):
            result.append(y[idx])
        elif isinstance(y, tuple):
            result.append(tuple(y_this[idx] for y_this in y))
        else:
            raise NotImplementedError

    return tuple(result)
