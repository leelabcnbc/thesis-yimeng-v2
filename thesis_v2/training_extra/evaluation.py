import numpy as np
import torch
from torch.nn.functional import mse_loss
from scipy.stats import pearsonr


# this is for computing loss function
def get_output_loss(*,
                    yhat, y, loss_type,
                    legacy=False,
                    handle_nan=False,
                    drop_all_nan_epoch=False,
                    ):
    assert not legacy
    # assert legacy
    if yhat.size() != y.size():
        assert yhat.size()[1:] == y.size()
        # broadcasting.
        y = y.view(1, *y.size()).expand(yhat.size())

    if handle_nan:
        good_vec = ~torch.isnan(y)
        if good_vec.sum().item() == 0:
            # ignore this epoch.
            assert drop_all_nan_epoch
            return 0.0
        y = y[good_vec]
        yhat = yhat[good_vec]

    if loss_type == 'mse':
        # return mse_loss(yhat, y)
        return mse_loss(yhat, y, reduction='mean')
    elif loss_type == 'poisson':
        # 1e-5 is for numerical stability.
        # same in NIPS2017 (mask CNN) code.
        # noinspection PyUnresolvedReferences
        return torch.mean(yhat - y * torch.log(yhat + 1e-5))
    else:
        raise NotImplementedError


def eval_fn_one_neuron(*, yhat, y, handle_nan=False, legacy_corr=True):
    assert legacy_corr
    if handle_nan:
        # ignore inf. here we assume yhat use nan to denote missing,
        # and inf might denote true data.
        good_vec = np.logical_not(np.isnan(y))
        y = y[good_vec]
        yhat = yhat[good_vec]

    return pearsonr(yhat, y)[0] if (np.std(yhat) > 1e-5 and np.std(y) > 1e-5) else 0


# this is for evaluation
def eval_fn_wrapper(*,
                    yhat_all, y_all, loss_type,
                    # setting return_corr to True can make tracking training performance a lot easier.
                    return_corr=True,
                    legacy_corr=True,
                    yhat_reduce_axis=0,
                    handle_nan=False,
                    ):
    # yhat_all and y_all
    # are batches of results as a list.
    # models always return list (even with only 1 element)
    yhat_all_neural = np.concatenate([x[0] for x in yhat_all], axis=yhat_reduce_axis)
    y_all_neural = np.concatenate([x[0] for x in y_all], axis=0)

    if yhat_all_neural.shape != y_all_neural.shape:
        assert yhat_all_neural.ndim == 3
        # if it's 3 dim, then first dim is timestep.
        # pick the last timestep.
        yhat_all_neural = yhat_all_neural[-1]

    assert yhat_all_neural.shape == y_all_neural.shape
    assert y_all_neural.ndim == 2

    assert legacy_corr
    # this is better, more stable (discarding small stds).
    # used in what and where NIPS paper's code.

    corr_each = np.array([
        eval_fn_one_neuron(
            yhat=yhat, y=y, handle_nan=handle_nan, legacy_corr=legacy_corr
        ) for yhat, y in zip(yhat_all_neural.T, y_all_neural.T)]
    )

    assert np.all(np.isfinite(corr_each))

    if len(y_all[0]) > 1:
        # not used yet (or ever).
        raise NotImplementedError
    else:
        acc = None
    # assert len(y_all[0]) == 1

    ret_dict = dict()

    if handle_nan:
        good_vec = np.logical_not(np.isnan(y_all_neural))
        yhat_all_neural = yhat_all_neural[good_vec]
        y_all_neural = y_all_neural[good_vec]

    if loss_type == 'poisson':
        ret_dict['loss_no_reg'] = float(
            np.mean(yhat_all_neural - y_all_neural * np.log(
                yhat_all_neural + 1e-5)))
    elif loss_type == 'mse':
        ret_dict['loss_no_reg'] = float(
            np.mean((yhat_all_neural - y_all_neural) ** 2)
        )
    else:
        raise NotImplementedError

    ret_dict.update(
        {
            'corr': corr_each.tolist() if return_corr else None,
            'corr_mean': float(corr_each.mean()),
            'corr_mean_neg': float(-corr_each.mean()),
            'corr2_mean': float((corr_each ** 2).mean()),
            'corr2_mean_neg': float(-(corr_each ** 2).mean()),
            'acc': acc,
        }
    )

    return ret_dict
