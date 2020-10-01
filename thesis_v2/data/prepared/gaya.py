import numpy as np
from modeling.data_utils import get_images, train_val_test_split
from analysis.data_utils import get_all_neural_data, spike_counts, trial_average, get_neural_data as get_neural_data_h

# legacy means the set up in
# https://github.com/leelabcnbc/gaya-data/blob/61f21849db0b195d95dda95b224b908206533026/modeling/scripts/train_data_driven_cnn.py   # noqa: E501

global_dict = {
    'tang_num_img': 2250,
    'tang_num_img_train': 1400,
    'tang_num_img_val': 350,
    'tang_num_img_test': 500,
    'tang_num_neuron': 34,
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


def images(dataset='both', crop=None):
    downsample = 4
    DATASET = dataset
    x_all = get_images(DATASET, downsample=downsample, torch_format=True,
                       normalize=False)
    if dataset == 'both':
        num_im_ref = global_dict['legacy_num_img']
    elif dataset == 'tang':
        num_im_ref = global_dict['tang_num_img']
    else:
        raise NotImplementedError

    assert x_all.shape == (num_im_ref, 1, global_dict['legacy_imsize'], global_dict['legacy_imsize'])

    if crop is not None:
        # crop the central crop x crop out of legacy_imsize
        slicer = slice((global_dict['legacy_imsize'] - crop) // 2, (global_dict['legacy_imsize'] + crop) // 2)
        x_all = x_all[:, :, slicer, slicer]
        assert x_all.shape == (num_im_ref, 1, crop, crop)

    assert x_all.min() >= 0
    assert x_all.max() <= 255
    return x_all


def get_neural_data(
        *,
        unit_mean_per_neuron=True,
        post_scale=None,
        start_offset=0,
        end_offset=100,
        dataset='both',
        return_raw=False,
):
    CORR_THRESHOLD = 0.7
    if dataset == 'both':
        y = get_all_neural_data(corr_threshold=CORR_THRESHOLD,
                                elecs=False)
    else:
        assert dataset in {'tang', 'googim'}
        y = get_neural_data_h(dataset=dataset, corr_threshold=CORR_THRESHOLD)
    # early response, for all the course project stuff
    y = spike_counts(y, start=540 + start_offset, end=540 + end_offset)
    # this is for computing ccnorm
    if return_raw:
        return y
    y = trial_average(y)

    if dataset == 'both':
        shape_ref = (
            global_dict['legacy_num_img'],
            global_dict['legacy_num_neuron']
        )
    elif dataset == 'tang':
        shape_ref = (
            global_dict['tang_num_img'],
            global_dict['tang_num_neuron']
        )
    else:
        raise NotImplementedError

    assert y.shape == shape_ref

    assert np.all(np.isfinite(y))

    if unit_mean_per_neuron:
        # this is the same type of processing done in 8k data
        per_neuron_mean = y.mean(axis=0)
        assert np.all(per_neuron_mean > 0)
        y = y / per_neuron_mean

    if post_scale is not None:
        y = y * post_scale

    assert y.shape == shape_ref

    return y


def get_indices(*, seed, dataset):
    assert seed == 'legacy'
    if dataset == 'both':
        total_size = global_dict['legacy_num_img']
        train_size = global_dict['legacy_num_img_train']
        val_size = global_dict['legacy_num_img_val']
        test_size = global_dict['legacy_num_img_test']
    elif dataset == 'tang':
        total_size = global_dict['tang_num_img']
        train_size = global_dict['tang_num_img_train']
        val_size = global_dict['tang_num_img_val']
        test_size = global_dict['tang_num_img_test']
    else:
        raise NotImplementedError

    assert total_size == train_size + val_size + test_size

    train_idx, val_idx, test_idx = train_val_test_split(
        total_size=total_size, train_size=train_size, val_size=val_size,
        deterministic=True
    )
    assert train_idx.shape == (train_size,)
    assert val_idx.shape == (val_size,)
    assert test_idx.shape == (test_size,)
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx, test_idx])),
                          np.arange(total_size))

    return train_idx, val_idx, test_idx


def get_data(*, seed, scale=None, dataset='both', start_offset=0, end_offset=100, crop=None):
    x_all = images(dataset=dataset, crop=crop)

    if dataset == 'both':
        num_im_ref = global_dict['legacy_num_img']
    elif dataset == 'tang':
        num_im_ref = global_dict['tang_num_img']
    else:
        raise NotImplementedError

    if crop is None:
        assert x_all.shape == (num_im_ref, 1, global_dict['legacy_imsize'], global_dict['legacy_imsize'])
    else:
        assert x_all.shape == (num_im_ref, 1, crop, crop)

    y = get_neural_data(post_scale=scale, dataset=dataset, start_offset=start_offset, end_offset=end_offset)

    indices = get_indices(seed=seed, dataset=dataset)

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
