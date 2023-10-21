import numpy as np
from os.path import join
from skimage.transform import downscale_local_mean, resize
from thesis_v2 import dir_dict
import matplotlib.pyplot as plt

global_dict = {
    'img_path': join(dir_dict['private_data'], 'tang_v1', 'images',),
    'rsp_path': join(dir_dict['private_data'], 'tang_v1', 'neural',),
    'num_img_train': {
        'm1s1': 34000,
        'm3s1': 34900
    },
    'num_img_val': 1000,
    'num_neurons_total': {
        'm1s1': 302,
        'm3s1': 324
    },
    # neurons with responses to
    'num_neurons_with_response_to_all_images': {
        'm1s1': 302,
        'm3s1': 324
    },
    'num_trial': 10,
}


def images(x, px_kept=80, final_size=40, force_resize=False):
    # assert px_kept == 80
    # assert final_size == 40
    slice_to_use = slice(50 - px_kept // 2, 50 + px_kept // 2)
    x_all = x[:, slice_to_use, slice_to_use]
    downscale_ratio = px_kept // final_size
    perfect_downscale = (downscale_ratio * final_size == px_kept)
    if perfect_downscale or (not force_resize):
        assert perfect_downscale
        scale_factors = (1, downscale_ratio, downscale_ratio)
        x_all = downscale_local_mean(x_all, scale_factors)[:, np.newaxis]
    else:
        print('use resize; not optimal for image quality')
        # force resize.
        x_all = np.asarray(
            [resize(x, (final_size, final_size), mode='edge',
                    anti_aliasing=True) for x in x_all]
        )[:, np.newaxis]

#     assert x_all.shape == (global_dict['num_img'], 1, final_size, final_size)
    assert x_all.min() >= 0
    assert x_all.max() <= 255
    return x_all


def get_neural_data(site, post_scale=None):
    y_train = np.load(join(global_dict['rsp_path'], f'trainRsp_{site}.npy'))
    y_val = np.load(join(global_dict['rsp_path'], f'valRsp_{site}.npy'))

    assert y_train.shape == (
        global_dict['num_img_train'][site],
        global_dict['num_neurons_with_response_to_all_images'][site]
    )
    assert np.all(np.isfinite(y_train))
    assert np.logical_not(np.any(np.all(np.isnan(y_train), axis=0), axis=0))

    assert y_val.shape == (
        global_dict['num_img_val'],
        global_dict['num_neurons_with_response_to_all_images'][site]
    )
    assert np.all(np.isfinite(y_val))
    assert np.logical_not(np.any(np.all(np.isnan(y_val), axis=0), axis=0))

    if post_scale is not None:
        y_train = y_train * post_scale
        y_val = y_val * post_scale
    return y_train, y_val

def get_neural_trials(site):
    y_val_trials = np.load(join(global_dict['rsp_path'], f'valRsp_trials_{site}.npy'))
    y_val_trials = y_val_trials.transpose((2, 0, 1))
    
    assert y_val_trials.shape == (
        global_dict['num_neurons_with_response_to_all_images'][site],
        global_dict['num_trial'],
        global_dict['num_img_val']
    )
    assert np.all(np.isfinite(y_val_trials))
    return y_val_trials


def get_data(*, site,
             px_kept,
             final_size,
             seed,
             scale=None,
             force_resize=False,
             ):
    img_train = np.load(join(global_dict['img_path'], f'trainPic_{site}.npy'))
    img_val = np.load(join(global_dict['img_path'], f'valPic_{site}.npy'))
    x_train = images(img_train, px_kept, final_size, force_resize=force_resize)
    x_val = images(img_val, px_kept, final_size, force_resize=force_resize)
    assert x_train.shape == (
        global_dict['num_img_train'][site], 1, final_size, final_size)
    assert x_val.shape == (
        global_dict['num_img_val'], 1, final_size, final_size)

    y_train, y_val = get_neural_data(site=site, post_scale=scale)

    result = [x_train, y_train, x_val, y_val, x_val, y_val]

    return tuple(result)
