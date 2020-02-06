from typing import List

import torch
from torch import nn

from .. import register_module, standard_init, bn_init_passthrough


class BLConvLayer(nn.Module):
    def __init__(self,
                 inchan: int,
                 outchan: int,
                 ksize: int,
                 bias: bool,
                 ):
        super().__init__()
        assert ksize % 2 == 1
        self.b_conv = nn.Conv2d(
            in_channels=inchan,
            out_channels=outchan,
            kernel_size=ksize,
            padding=ksize // 2,
            bias=bias,
        )

        self.l_conv = nn.Conv2d(
            in_channels=outchan,
            out_channels=outchan,
            kernel_size=ksize,
            padding=ksize // 2,
            bias=bias,
        )

    def forward(self, b_input, l_input, bias_output=None):
        b_output = None
        l_output = None

        if b_input is not None:
            b_output = self.b_conv(b_input)
        if l_input is not None:
            l_output = self.l_conv(l_input)

        if b_output is not None and l_output is not None:
            sum_output = b_output + l_output
        elif b_output is None and l_output is not None:
            sum_output = l_output
        elif b_output is not None and l_output is None:
            sum_output = b_output
        else:
            raise RuntimeError('at least one source is not None')

        if bias_output is not None:
            sum_output = sum_output + bias_output

        return sum_output


class BLConvLayerStack(nn.Module):
    def __init__(self,
                 *,
                 n_timesteps: int,
                 channel_list: List[int],
                 ksize_list: List[int],
                 bias: bool = False,
                 act_fn: str = 'relu',
                 # these two values match those set in `thesis_v2/blocks_json/general.py`
                 bn_eps=0.001,
                 bn_momentum=0.1,
                 pool_ksize=2,
                 ):
        # channel_list should be of length 1+number of layers.
        # channel_list[0] being the number of channels for input
        super().__init__()
        self.n_timesteps = n_timesteps

        n_layer = len(ksize_list)
        assert n_layer >= 1 and len(channel_list) == 1 + n_layer
        self.n_layer = n_layer

        self.layer_list = nn.ModuleList(
            [BLConvLayer(inchan=channel_list[i],
                         outchan=channel_list[i + 1],
                         ksize=ksize_list[i],
                         bias=bias) for i in range(n_layer)]
        )

        # BN layers.
        self.bn_layer_list = []
        for t in range(n_timesteps):
            # https://discuss.pytorch.org/t/convering-a-batch-normalization-layer-from-tf-to-pytorch/20407/2
            self.bn_layer_list.extend([nn.BatchNorm2d(num_features=channel_list[i + 1],
                                                      eps=bn_eps, momentum=bn_momentum) for i in range(n_layer)])
        self.bn_layer_list = nn.ModuleList(
            self.bn_layer_list
        )

        if act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act_fn == 'softplus':
            self.act_fn = nn.Softplus()
        elif act_fn is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError

        self.pool_ksize = pool_ksize
        assert self.pool_ksize >= 1

        if self.pool_ksize > 1:
            self.pool = nn.MaxPool2d(kernel_size=pool_ksize, ceil_mode=True)
        else:
            self.pool = nn.Identity()

    def forward(self, b_input):
        # main loop
        last_out = [None for _ in range(self.n_layer)]

        # cache first layer's first time output.
        first_layer_first_time_output = None

        output_list = []

        for t in range(self.n_timesteps):
            for layer_idx in range(self.n_layer):
                layer_this = self.layer_list[layer_idx]
                bn_this = self.bn_layer_list[t * self.n_layer + layer_idx]
                if layer_idx == 0:
                    if t == 0:
                        first_layer_first_time_output = layer_this(b_input, None)
                        last_out[layer_idx] = first_layer_first_time_output
                    else:
                        last_out[layer_idx] = layer_this(None, last_out[layer_idx], first_layer_first_time_output)
                else:
                    pooled_input: torch.Tensor = self.pool(last_out[layer_idx - 1])
                    last_out[layer_idx] = layer_this(pooled_input, last_out[layer_idx])

                # do batch norm
                last_out[layer_idx] = bn_this(last_out[layer_idx])

                # do act
                last_out[layer_idx] = self.act_fn(last_out[layer_idx])

            output_list.append(last_out[self.n_layer - 1])

        # return a tuple of Tensors, of length `self.n_timesteps`.
        return tuple(output_list)


class RecurrentAccumulator(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        assert mode in {'instant', 'cummean'}
        self.mode = mode

    def forward(self, input_tensor_tuple):
        assert isinstance(input_tensor_tuple, tuple)
        ret = []
        if self.mode == 'instant':
            # instant readout mode.
            return input_tensor_tuple
        elif self.mode == 'cummean':
            # this is the cumulative mode in the original paper.
            # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/3
            for i in range(len(input_tensor_tuple)):
                ret.append(torch.mean(torch.stack(input_tensor_tuple[:i + 1]), 0))
            return tuple(ret)
        else:
            raise ValueError


# pcn local init
def blconvlayerstack_init(mod: BLConvLayerStack, init: dict) -> None:
    n_time = mod.n_timesteps
    n_layer = mod.n_layer

    attrs_to_init = [
                        f'layer_list.{x}.b_conv.weight' for x in range(n_layer)
                    ] + [
                        f'layer_list.{x}.l_conv.weight' for x in range(n_layer)
                    ]
    attrs_to_init_zero_optional = [
                                      f'layer_list.{x}.b_conv.bias' for x in range(n_layer)
                                  ] + [
                                      f'layer_list.{x}.l_conv.bias' for x in range(n_layer)
                                  ]
    left_out_attrs = [
                         f'bn_layer_list.{x}.weight' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.bias' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.num_batches_tracked' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.running_var' for x in range(n_time * n_layer)
                     ]+ [
                         f'bn_layer_list.{x}.running_mean' for x in range(n_time * n_layer)
                     ]


    # all bns
    for i in range(n_time * n_layer):
        bn_init_passthrough(
            mod.bn_layer_list[i], dict()
        )

    # all ff convs
    # all lateral convs
    standard_init(
        mod, init, attrs_to_init=tuple(attrs_to_init),
        attrs_to_init_zero_optional=tuple(attrs_to_init_zero_optional),
        left_out_attrs=tuple(left_out_attrs)
    )


register_module('rcnn_kriegeskorte.blstack', BLConvLayerStack, blconvlayerstack_init)
register_module('rcnn_kriegeskorte.accumulator', RecurrentAccumulator)
