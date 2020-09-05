import numpy as np
from skimage.transform import resize

import torch

from analysis.data_utils import DATA_DIR
from modeling.data_utils import torchvision_normalize

# check `/scripts/debug/hal_analysis/tuning_analysis_debug.ipynb` on some sanity check of stimuli and bars
def get_bars():
    # 0/180
    horz_bar = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]])
    # 22.5
    horz_r_bar = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0]])
    # 45
    rdia_bar = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]])
    # 67.5
    vert_r_bar = np.array([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]])
    # 90
    vert_bar = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]])
    # 112.5
    vert_l_bar = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]])
    # 135
    ldia_bar = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]])
    # 167.5
    horz_l_bar = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0]])
    # you must follow this order.
    # this order is consistent with order in `get_stimuli_dict`
    bars = [horz_bar, horz_r_bar, rdia_bar, vert_r_bar, vert_bar,
            vert_l_bar, ldia_bar, horz_l_bar]

    bars = [bar / np.linalg.norm(bar.ravel()) for bar in bars]
    return bars


def get_stimuli(*, num_channel, normalize, new_size):
    ot_stimuli: np.ndarray = np.load(f'{DATA_DIR}/misc/tang_ot.npy')
    assert ot_stimuli.shape == (320, 160, 160) and ot_stimuli.min() >= 0.0 and ot_stimuli.max() <= 255.0
    if new_size is not None:
        # then resize each one
        # print('use resize; not optimal for image quality')
        # force resize.
        ot_stimuli = np.asarray(
            [resize(x, (new_size, new_size), mode='edge', anti_aliasing=True) for x in ot_stimuli]
        )
    else:
        new_size = 160
    ot_stimuli = ot_stimuli[:, np.newaxis]
    assert ot_stimuli.shape == (320, 1, new_size, new_size)

    if num_channel != 1:
        # broadcast
        ot_stimuli = np.broadcast_to(ot_stimuli, (320, num_channel, new_size, new_size))

    if normalize:
        ot_stimuli = torchvision_normalize(ot_stimuli)
    return ot_stimuli


def get_stimuli_dict(*, num_channel=1, normalize=False, new_size=None):
    edge_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
                                      for k in range(0, 2)]).flatten() for j in range(8)])
    bar_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
                                     for k in range(2, 6)]).flatten() for j in range(8)])
    hatch_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
                                       for k in range(6, 8)]).flatten() for j in range(8)])

    # sanity check
    assert np.array_equal(
        np.sort(np.concatenate([edge_o_idxs.ravel(), bar_o_idxs.ravel(), hatch_o_idxs.ravel()])),
        np.arange(320)
    )

    assert edge_o_idxs.shape[0] == bar_o_idxs.shape[0] == hatch_o_idxs.shape[0] == 8

    # each of `_idx` is arranged into a (8, x) matrix, i'th (0-indexed_ row with stimuli of orientation `i*22.5`
    return {
        'idx_dict': {
            'edge': edge_o_idxs,
            'bar': bar_o_idxs,
            'hatch': hatch_o_idxs,
        },
        'stimuli': get_stimuli(num_channel=num_channel, normalize=normalize, new_size=new_size)
    }


def get_tunings_dict(acts, idx_dict):
    num_im, num_c = acts.shape
    assert np.array_equal(
        np.sort(np.concatenate(list(x.ravel() for x in idx_dict.values()))),
        np.arange(num_im)
    )

    tunings_dict = dict()
    for k, idx_this in idx_dict.items():
        assert idx_this.ndim == 2
        tuning_this = np.zeros((num_c, idx_this.shape[0]))
        for idx_or, idx_imgs in enumerate(idx_this):
            # get max response overall all images, per channel
            acts_this = acts[idx_imgs].max(axis=0)
            assert acts_this.shape == (num_c,)
            tuning_this[:, idx_or] = acts_this
        tunings_dict[k] = tuning_this

    return tunings_dict


def get_tuning_diffs(*, num_c, self_weights, tunings_dict, bars, kernel_shape=(3, 3), num_orientation=8):
    # go through the kernels
    goods = np.zeros((num_c,))
    bads = np.zeros((num_c,))

    assert num_orientation % 2 == 0

    for b in bars:
        assert b.shape == kernel_shape

    for v in tunings_dict.values():
        assert v.shape == (num_c, num_orientation)

    for i in range(num_c):
        # normalize weight
        n_weight = self_weights[i] / np.linalg.norm(self_weights[i].ravel())
        assert n_weight.shape == kernel_shape

        # extent to which weight aligns with each direction
        projs = [(n_weight.ravel()) @ (bar.ravel()) for bar in bars]

        tunings = [vv[i] for vv in tunings_dict.values()]

        ranges = [x.max() - x.min() for x in tunings]
        best: int = np.argmax(ranges)

        best_range = tunings[best]
        assert best_range.shape == (num_orientation, )
        best_orient: int = np.argmax(best_range)
        orthogonal_orient = (best_orient + num_orientation // 2) % num_orientation
        goods[i] = projs[best_orient]
        bads[i] = projs[orthogonal_orient]

    return {
        'goods': goods,
        'bads': bads,
        # this measures correlation between `projs` and `tunings`
        'diffs': goods - bads,
    }


def model_orientation_tuning_one(*, model, get_resp_fn, get_self_weights_fn, stimuli_dict, bars):
    # here, model gives a 320 x num_channel x H x W response map, given the (320, 1, new_size, new_size) ot_stimuli
    # get responses.
    stimuli = stimuli_dict['stimuli']

    # just evaluate in CPU, for simplicity.
    with torch.no_grad():
        acts = get_resp_fn(model, torch.tensor(stimuli, dtype=torch.float32))

        if isinstance(acts, torch.Tensor):
            acts = acts.numpy()

    assert type(acts) is np.ndarray

    # print(acts.mean(), acts.std(), acts.shape)
    # get central column
    num_im, num_c, h, w = acts.shape
    acts = acts[:, :, h // 2, w // 2]

    tunings_dict = get_tunings_dict(acts, stimuli_dict['idx_dict'])
    self_weights = get_self_weights_fn(model)

    tuning_diffs = get_tuning_diffs(
        num_c=num_c,
        self_weights=self_weights,
        tunings_dict=tunings_dict,
        bars=bars
    )

    return tuning_diffs
