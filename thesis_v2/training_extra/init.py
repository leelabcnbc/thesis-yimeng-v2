import torch
import numpy as np

from torch.nn import Softplus, ReLU


def check_init_type(final_act_module):
    if isinstance(final_act_module, Softplus):
        init_type = 'softplus'
    elif isinstance(final_act_module, ReLU) or final_act_module is None:
        init_type = 'none'
    else:
        raise NotImplementedError
    return init_type


def init_bias_wrapper(bias: torch.Tensor, mean_response: np.ndarray,
                      *, init_type):
    # hacked from
    # https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/thesis_proposal/cnn.py#L481-L496  # noqa: E501

    if init_type == 'softplus':
        b = inv_softplus(mean_response)
    elif init_type == 'none':
        b = mean_response
    elif init_type == 'exp':
        b = np.log(mean_response)
    else:
        raise NotImplementedError
    assert b.shape == bias.size()
    assert np.all(np.isfinite(b))
    with torch.no_grad():
        # noinspection PyCallingNonCallable
        bias[()] = torch.tensor(b)


def inv_softplus(x):
    # from https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/thesis_proposal/cnn.py#L255-L259  # noqa: E501
    assert np.all(x > 0)
    # copied from original code.
    # I think numerically it's not very stable.
    return np.log(np.exp(x) - 1)
