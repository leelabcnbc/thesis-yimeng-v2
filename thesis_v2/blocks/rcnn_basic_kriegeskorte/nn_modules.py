from typing import List

import torch
from torch import nn


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
                 bn_eps=1e-5,
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
                                                      eps=bn_eps) for i in range(n_layer)])
        self.bn_layer_list = nn.ModuleList(
            self.bn_layer_list
        )

        if act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act_fn == 'softplus':
            self.act_fn = nn.Softplus()
        else:
            raise NotImplementedError

        self.pool = nn.MaxPool2d(kernel_size=2)

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

        # return a list of Tensors, of length `self.n_timesteps`.
        return output_list


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
