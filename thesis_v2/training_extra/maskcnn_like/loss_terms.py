"""loss terms for the CNN in NIPS 2017 paper

Neural system identification for large populations separating "what" and "where"  # noqa: E501
https://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where  # noqa: E501
"""

from typing import List

import torch
from torch import nn
from torch.nn.functional import conv2d


def maskcnn_loss_v1_weight_readout(module, *, scale, legacy):
    if scale == 0:
        return 0
    weight_feature_flat = module.weight_feature.view(
        module.weight_feature.size()[0], -1)
    weight_spatial_flat = module.weight_spatial.view(
        module.weight_spatial.size()[0], -1)
    assert not legacy
    # noinspection PyUnresolvedReferences
    return scale * torch.mean(
        torch.sum(torch.abs(weight_feature_flat), 1) *
        torch.sum(torch.abs(weight_spatial_flat), 1)
    )


def maskcnn_loss_v1_kernel_group_sparsity(module_list: List[nn.Conv2d],
                                          scale_list: list):
    """group_sparsity_regularizer_2d in original code"""
    # basically, sum up each HxW slice individually.
    sum_list = []
    for m, s in zip(module_list, scale_list):
        if s == 0:
            continue
        w_this: nn.Parameter = m.weight
        c_out, c_in, h, w = w_this.size()
        w_this = w_this.view(c_out, c_in, h * w)
        # noinspection PyUnresolvedReferences
        sum_to_add = s * torch.sum(torch.sqrt(torch.sum(w_this ** 2, -1)))
        sum_list.append(sum_to_add)
    return sum(sum_list)


_maskcnn_loss_v1_kernel_smoothness_kernel = {'data': None}


def maskcnn_loss_v1_kernel_smoothness(module_list: List[nn.Conv2d],
                                      scale_list: list, device=None):
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device is not None
    if _maskcnn_loss_v1_kernel_smoothness_kernel['data'] is None:
        # noinspection PyCallingNonCallable, PyUnresolvedReferences
        _maskcnn_loss_v1_kernel_smoothness_kernel['data'] = torch.tensor(
            [[[[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]]],
            dtype=torch.float32
        ).to(device)
        # 1 x 1 x 3 x 3
    kernel = _maskcnn_loss_v1_kernel_smoothness_kernel['data']

    """group_sparsity_regularizer_2d in original code"""
    # basically, sum up each HxW slice individually.
    sum_list = []
    for m, s in zip(module_list, scale_list):
        if s == 0:
            continue
        w_this: nn.Parameter = m.weight
        c_out, c_in, h, w = w_this.size()

        w_this = w_this.view(c_out * c_in, 1, h, w)
        w_this_conved = conv2d(w_this, kernel, padding=1).view(c_out, c_in, -1)
        w_this = w_this.view(c_out, c_in, -1)

        # noinspection PyUnresolvedReferences
        sum_to_add = s * torch.sum(
            torch.sum(w_this_conved ** 2, -1) / torch.sum(w_this ** 2, -1))
        sum_list.append(sum_to_add)
    return sum(sum_list)
