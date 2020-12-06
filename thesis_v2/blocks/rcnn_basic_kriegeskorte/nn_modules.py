from typing import List, Optional
from collections import Counter
import torch
from torch import nn

from .. import register_module, standard_init, bn_init_passthrough
from ...analysis.utils import get_source_analysis_for_one_model_spec, LayerSourceAnalysis, extract_l_and_t


class BLConvLayer(nn.Module):
    def __init__(self,
                 inchan: int,
                 outchan: int,
                 ksize: int,
                 bias: bool,
                 generate_lateral: bool,
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
        ) if generate_lateral else None

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


def extract_l_and_type(s):
    t = s[0]
    assert t in {'R', 'B'}
    return int(s[1:]) - 1, t


def get_bn_counter(multipath_source: List[LayerSourceAnalysis]):
    counter = Counter()
    for src_analysis in multipath_source:
        for src_dict in src_analysis.source_list:
            counter.update(extract_l_and_t(x) for x in src_dict['conv'] if x.startswith('s'))
    return counter


def verify_unique_path(multipath_source, n_timesteps):
    assert len(multipath_source) == n_timesteps
    for timestep_idx, chain_list_this in enumerate(multipath_source):
        conv_list = [x['conv'] for x in chain_list_this.source_list]
        assert len(conv_list) == len(set(conv_list))


def get_depths(multipath_source, n_timesteps, num_recurrent_layer):
    depth_set = set()
    assert len(multipath_source) == n_timesteps
    for chain_list_this in multipath_source:
        for conv in chain_list_this.source_list:
            assert conv['conv'][0] == 'I'
            assert len(conv['conv'][1:]) > 0
            assert len(conv['conv'][1:]) % 2 == 0
            depth_set.add(len(conv['conv'][1:]) // 2)

    assert num_recurrent_layer in {1,2}
    assert depth_set == set(range(num_recurrent_layer, n_timesteps+num_recurrent_layer))
    return depth_set


def update_allowed_depth(allowed_depth: set, spec):
    # first type of hack spec: leDX
    # where le means <=, D means depth X is layer depth.
    # second type: geDX
    # ge means >=
    # third type (not implemented yet)
    # onlyDX1,X2,X3,...
    # only keeps paths of depth X1,X2,X3,....

    # note that leDX is NOT the same as reducing number of time steps to X.
    # because each time step has paths of depths shorter than the time.
    # therefore, for a 7-iteration model, leD3 will still emit results across 7 iterations.

    # I need to check that the depth makes sense.
    if spec.startswith('leD'):
        depth_threshold = int(spec[3:])
        assert depth_threshold in allowed_depth
        ret = {d for d in allowed_depth if d <= depth_threshold}
    elif spec.startswith('geD'):
        depth_threshold = int(spec[3:])
        assert depth_threshold in allowed_depth
        ret = {d for d in allowed_depth if d >= depth_threshold}
    elif spec.startswith('onlyD'):
        raise NotImplementedError
    else:
        raise ValueError

    return ret


class BLConvLayerStack(nn.Module):
    def __init__(self,
                 *,
                 n_timesteps: int,
                 channel_list: List[int],
                 ksize_list: List[int],
                 # = 'relu'
                 act_fn: str,
                 # these two values match those set in `thesis_v2/blocks_json/general.py`
                 bn_eps=0.001,
                 bn_momentum=0.1,
                 # =2,
                 pool_ksize,
                 # ='max'
                 pool_type,
                 bias: bool = False,
                 norm_type: str = 'batchnorm',
                 # multi-path ensemble
                 multi_path: bool = False,
                 # if `multi_path_separate_bn` is true, each path has its own BNs;
                 # otherwise, they share some BNs.
                 multi_path_separate_bn: Optional[bool] = None,
                 multi_path_hack: Optional[str] = None,
                 ):
        # channel_list should be of length 1+number of layers.
        # channel_list[0] being the number of channels for input
        super().__init__()
        assert not bias
        self.n_timesteps = n_timesteps

        n_layer = len(ksize_list)
        assert n_layer >= 1 and len(channel_list) == 1 + n_layer
        self.n_layer = n_layer

        self.layer_list = nn.ModuleList(
            [BLConvLayer(inchan=channel_list[i],
                         outchan=channel_list[i + 1],
                         ksize=ksize_list[i],
                         bias=bias,
                         generate_lateral=(n_timesteps > 1)) for i in range(n_layer)]
        )

        self.multi_path = multi_path
        self.multi_path_separate_bn = multi_path_separate_bn
        self.multi_path_hack = multi_path_hack

        if self.multi_path:
            self.multipath_source = get_source_analysis_for_one_model_spec(
                num_recurrent_layer=n_layer, num_cls=n_timesteps,
                # any one is fine.
                readout_type='inst-last',
                return_raw=True,
                add_bn_in_chain=True
            )[-1]
            # print([x.source_list for x in self.multipath_source])
            # assert each each list has unique chains of convs
            # THIS IS IMPORTANT, because it's possible that I double counted some path during eval.
            verify_unique_path(self.multipath_source, self.n_timesteps)
            assert type(self.multi_path_separate_bn) is bool
            self.allowed_depth = get_depths(
                self.multipath_source,
                n_timesteps=n_timesteps,
                num_recurrent_layer=n_layer,
            )
            # check multi_path_hack's value makes sense
            if self.multi_path_hack is not None:
                assert type(self.multi_path_hack) is str
                self.allowed_depth = update_allowed_depth(self.allowed_depth, self.multi_path_hack)

        else:
            self.multipath_source = None
            assert self.multi_path_separate_bn is None
            assert self.multi_path_hack is None

        # BN layers.
        self.bn_layer_list = []
        if norm_type == 'batchnorm':
            for t in range(n_timesteps):
                # https://discuss.pytorch.org/t/convering-a-batch-normalization-layer-from-tf-to-pytorch/20407/2
                self.bn_layer_list.extend([nn.BatchNorm2d(num_features=channel_list[i + 1],
                                                          eps=bn_eps, momentum=bn_momentum) for i in range(n_layer)])
        elif norm_type == 'instancenorm':
            # instance norm layers
            for t in range(n_timesteps):
                self.bn_layer_list.extend([nn.InstanceNorm2d(num_features=channel_list[i + 1],
                                                             eps=bn_eps, momentum=bn_momentum, affine=True) for i in
                                           range(n_layer)])
            # first, all layers for t=0
            # then t=1,
            # then t=2,
            # ...
            # then t=len(n_timesteps)
        else:
            raise ValueError

        self.bn_layer_list = nn.ModuleList(
            self.bn_layer_list
        )

        if self.multi_path_separate_bn:
            # then we need to create additional copies of BNs
            # probably the easist way is to create some ModuleList variables called
            # self.bn_layer_add_list_{layer_idx}_{time_idx}
            # using setattr.
            # when obtaining them, use getattr
            # the reason for using this hack, compared to Dict->List is because a regular dict cannot be
            # properly registered, and ModuleDict is supposed to only work with Module typed values, rather
            # than ModuleList.

            # not needed for instance norm as that performed worse.
            assert norm_type == 'batchnorm'
            # use a counter to count.
            counter: Counter = get_bn_counter(self.multipath_source)
            for (layer_idx, time_idx), ct in counter.items():
                # not [BatchNorm2d]*(ct-1), which duplicates BN layers.
                if ct > 1:
                    bn_list_this = nn.ModuleList([
                        nn.BatchNorm2d(num_features=channel_list[layer_idx + 1], eps=bn_eps, momentum=bn_momentum) for _
                        in range(ct - 1)
                    ])
                    setattr(self, f'bn_layer_add_list_{layer_idx}_{time_idx}', bn_list_this)
                else:
                    # no need for additional. just use the original one.
                    pass
            self.add_bn_counter = counter
        else:
            self.add_bn_counter = None

        # to capture intermediate response.
        self.capture_list = nn.ModuleList(
            [nn.Identity() for _ in range(self.n_layer)]
        )

        # capture input
        self.input_capture = nn.Identity()

        if act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act_fn == 'softplus':
            self.act_fn = nn.Softplus()
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act_fn is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError

        self.pool_ksize = pool_ksize
        assert self.pool_ksize >= 1
        self.pool_type = pool_type

        if self.pool_ksize > 1:
            if self.pool_type == 'max':
                self.pool = nn.MaxPool2d(kernel_size=pool_ksize, ceil_mode=True)
            elif self.pool_type == 'avg':
                self.pool = nn.AvgPool2d(kernel_size=pool_ksize, ceil_mode=True)
            else:
                raise ValueError
        else:
            assert self.pool_type is None
            self.pool = nn.Identity()

    def obtain_conv(self, comp):
        layer, t = extract_l_and_type(comp)
        layer_this: BLConvLayer = self.layer_list[layer]
        if t == 'B':
            return layer_this.b_conv
        elif t == 'R':
            return layer_this.l_conv
        else:
            raise ValueError

    def obtain_bn(self, comp, counter: Counter, return_layer_time=False):
        layer, time = extract_l_and_t(comp)
        if (not self.multi_path_separate_bn) or (counter[layer, time] == 0):
            # use the original one.
            bn_this = self.bn_layer_list[time * self.n_layer + layer]
        else:
            # use later one.
            bn_this = getattr(self, f'bn_layer_add_list_{layer}_{time}')[counter[layer, time] - 1]
        counter[layer, time] += 1
        if not return_layer_time:
            return bn_this
        else:
            return bn_this, (layer, time)

    def evaluate_multi_path(self, b_input):
        # counter tracks how many times a BN layer is used.
        counter = Counter()
        output_list = []
        # I do this just to get typing.
        multipath_source: List[LayerSourceAnalysis] = self.multipath_source
        assert len(multipath_source) == self.n_timesteps
        for timestep_idx, chain_list_this in enumerate(multipath_source):
            output_list_this_time = []
            for chain_this in chain_list_this.source_list:
                # pairs of conv and BN
                chain_this_raw = chain_this['conv'][1:]

                depth_this = len(chain_this_raw) // 2
                if depth_this not in self.allowed_depth:
                    continue

                # create sequence. note that BN has test vs train. this is encapsulated in BN itself.
                # we don't need to handle it explicitly.
                # check the implementatinon nn.Sequential. DO NOT use Sequential directly as the Sequential's
                # train/test mode is inconsistent with BN's.
                output_now = b_input
                for idx, comp in enumerate(chain_this_raw):
                    # obtain the corresponding component.
                    if idx % 2 == 0:
                        mod = self.obtain_conv(comp)
                    else:
                        mod = self.obtain_bn(comp, counter)
                    output_now = mod(output_now)
                    if idx % 2 == 1:
                        # end of a pair. apply act fn
                        output_now = self.act_fn(output_now)
                output_list_this_time.append(output_now)
            # sum together all tensors in the currentcollect_rcnn_k_bl_main_result timestamp.
            if len(output_list_this_time) > 0:
                output_list.append(
                    torch.sum(torch.stack(output_list_this_time), 0)
                )
        assert len(output_list) > 0
        return output_list

    def forward(self, b_input):
        # capture
        b_input = self.input_capture(b_input)
        if not self.multi_path:
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

                    # capture
                    last_out[layer_idx] = self.capture_list[layer_idx](last_out[layer_idx])

                output_list.append(last_out[self.n_layer - 1])
        else:
            output_list = self.evaluate_multi_path(b_input)

        # return a tuple of Tensors, of length `self.n_timesteps`.
        return tuple(output_list)


class RecurrentAccumulator(nn.Module):
    def __init__(self, mode: str, drop: int = 0, order: int = 1):
        super().__init__()
        # `last` is essentially `instant_last`
        assert mode in {'instant', 'cummean', 'last', 'cummean_last'}
        self.mode = mode
        assert drop >= 0
        self.drop = drop
        self.order = order

    def acc_mean_inner(self, input_tensor_tuple, order: int):
        ret = []
        # this is the cumulative mode in the original paper.
        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/3
        for i in range(len(input_tensor_tuple)):
            ret.append(torch.mean(torch.stack(input_tensor_tuple[:i + 1]), 0))
        if order == 1:
            return tuple(ret)
        else:
            return self.acc_mean_inner(ret, order - 1)

    def forward(self, input_tensor_tuple):
        assert isinstance(input_tensor_tuple, tuple)
        # drop first `drop`
        input_tensor_tuple = input_tensor_tuple[self.drop:]
        if self.mode == 'instant':
            # instant readout mode.
            return input_tensor_tuple
        elif self.mode == 'cummean':
            return self.acc_mean_inner(input_tensor_tuple, self.order)
        elif self.mode == 'cummean_last':
            # this is probably closet to the actual evaluation protocol (using the last)
            return self.acc_mean_inner(input_tensor_tuple, self.order)[-1:]
        elif self.mode == 'last':
            return input_tensor_tuple[-1:]
        else:
            raise ValueError


# pcn local init
def blconvlayerstack_init(mod: BLConvLayerStack, init: dict) -> None:
    n_time = mod.n_timesteps
    n_layer = mod.n_layer

    attrs_to_init = [
                        f'layer_list.{x}.b_conv.weight' for x in range(n_layer)
                    ] + ([
                             f'layer_list.{x}.l_conv.weight' for x in range(n_layer)
                         ] if n_time > 1 else [])
    attrs_to_init_zero_optional = [
                                      f'layer_list.{x}.b_conv.bias' for x in range(n_layer)
                                  ] + ([
                                           f'layer_list.{x}.l_conv.bias' for x in range(n_layer)
                                       ] if n_time > 1 else [])
    left_out_attrs = [
                         f'bn_layer_list.{x}.weight' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.bias' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.num_batches_tracked' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.running_var' for x in range(n_time * n_layer)
                     ] + [
                         f'bn_layer_list.{x}.running_mean' for x in range(n_time * n_layer)
                     ]

    # all bns
    for i in range(n_time * n_layer):
        # this should work for both BN and instance norm, because instance norm is
        # derived from batch norm
        bn_init_passthrough(
            mod.bn_layer_list[i], dict()
        )

    if mod.add_bn_counter is not None:
        # initialize all additional BN stuffs.
        counter = mod.add_bn_counter
        for (layer_idx, time_idx), ct in counter.items():
            # not [BatchNorm2d]*(ct-1), which duplicates BN layers.
            if ct > 1:
                mod_list_name = f'bn_layer_add_list_{layer_idx}_{time_idx}'
                mod_list = getattr(mod, mod_list_name)
                left_out_attrs += (
                        [f'{mod_list_name}.{x}.weight' for x in range(ct - 1)] +
                        [f'{mod_list_name}.{x}.bias' for x in range(ct - 1)] +
                        [f'{mod_list_name}.{x}.num_batches_tracked' for x in range(ct - 1)] +
                        [f'{mod_list_name}.{x}.running_var' for x in range(ct - 1)] +
                        [f'{mod_list_name}.{x}.running_mean' for x in range(ct - 1)]
                )
                # initialize
                for ct_idx in range(ct - 1):
                    bn_init_passthrough(mod_list[ct_idx], dict())

    # all ff convs
    # all lateral convs
    standard_init(
        mod, init, attrs_to_init=tuple(attrs_to_init),
        attrs_to_init_zero_optional=tuple(attrs_to_init_zero_optional),
        left_out_attrs=tuple(left_out_attrs)
    )


register_module('rcnn_kriegeskorte.blstack', BLConvLayerStack, blconvlayerstack_init)
register_module('rcnn_kriegeskorte.accumulator', RecurrentAccumulator)
