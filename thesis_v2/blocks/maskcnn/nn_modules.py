"""custom (C)NN modules"""
from typing import Union, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional
import math

from ...blocks_json.utils import check_input_size
from .. import register_module, standard_init


class FactoredLinear2D(nn.Module):
    """
    skeleton copied from implementation of nn.Linear from PyTorch 0.3.1
    """

    def __init__(self, in_channels: int,
                 map_size: Union[int, Tuple[int, int]],
                 out_features: int, bias: bool = True,
                 weight_feature_constraint: Optional[str] = None,
                 weight_spatial_constraint: Optional[str] = None) -> None:
        super().__init__()
        assert isinstance(in_channels, int) and in_channels > 0
        self.in_channels = in_channels

        self.map_size = check_input_size(map_size)

        assert isinstance(out_features, int) and out_features > 0
        self.out_features = out_features

        assert weight_feature_constraint in {None, 'abs'}
        self.weight_feature_constraint = weight_feature_constraint
        assert weight_spatial_constraint in {None, 'abs'}
        self.weight_spatial_constraint = weight_spatial_constraint

        # use torch.empty as it's the preferred way to create tensors
        # in PyTorch 0.4, instead of torch.Tensor.
        # noinspection PyUnresolvedReferences
        self.weight_spatial: nn.Parameter = nn.Parameter(
            torch.empty(self.out_features, self.map_size[0], self.map_size[1]))
        # noinspection PyUnresolvedReferences
        self.weight_feature: nn.Parameter = nn.Parameter(
            torch.empty(self.out_features, self.in_channels))
        if bias:
            # noinspection PyUnresolvedReferences
            self.bias: nn.Parameter = nn.Parameter(
                torch.empty(self.out_features))
        else:
            # noinspection PyTypeChecker
            self.register_parameter('bias', None)
        self.reset_parameters()
        # print('changed impl')

    def reset_parameters(self) -> None:
        # this is simply adapted from nn.Linear. should always be initialized
        # by hand.
        stdv = 1. / math.sqrt(
            self.in_channels * self.map_size[0] * self.map_size[1])
        # usage of `data` is discouraged in PyTorch 0.4. but let's just stick
        # to it for simplicity.
        self.weight_spatial.data.uniform_(-stdv, stdv)
        self.weight_feature.data.fill_(1.0)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x: torch.Tensor):
        # I assume that input has shape (N, in_channels, map_size[0],
        # map_size[1]
        # first get the weights.

        weight_spatial_view = self.weight_spatial
        weight_feature_view = self.weight_feature

        if self.weight_feature_constraint is not None:
            if self.weight_feature_constraint == 'abs':
                # weight_spatial_view = torch.abs(weight_spatial_view)
                # noinspection PyUnresolvedReferences
                weight_feature_view = torch.abs(weight_feature_view)
            else:
                raise RuntimeError

        if self.weight_spatial_constraint is not None:
            if self.weight_spatial_constraint == 'abs':
                # noinspection PyUnresolvedReferences
                weight_spatial_view = torch.abs(weight_spatial_view)
            else:
                raise RuntimeError

        weight_spatial_view = weight_spatial_view.view(self.out_features, 1,
                                                       self.map_size[0],
                                                       self.map_size[1])
        weight_feature_view = weight_feature_view.view(self.out_features,
                                                       self.in_channels, 1, 1)

        # then broadcast to get new weight.
        if self.in_channels != 1:
            weight = weight_spatial_view * weight_feature_view
        else:
            # feature weighting not needed
            # this is for both quicker learning, as well as being compatible
            # with `CNN.py` in the original repo.
            weight = weight_spatial_view.expand(self.out_features,
                                                self.in_channels,
                                                self.map_size[0],
                                                self.map_size[1])
        weight = weight.view(self.out_features,
                             self.in_channels * self.map_size[0] *
                             self.map_size[1])
        return functional.linear(input_x.view(input_x.size(0), -1),
                                 weight, self.bias)


def factorfc_init(mod: FactoredLinear2D, init: dict) -> None:
    # this will make
    standard_init(mod, init, attrs_to_init=('weight_spatial',
                                            'weight_feature'))


register_module('maskcnn.factoredfc', FactoredLinear2D, factorfc_init)
