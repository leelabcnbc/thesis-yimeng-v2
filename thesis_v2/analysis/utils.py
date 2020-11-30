from collections import defaultdict, Counter
from functools import reduce
from copy import deepcopy


class LayerSourceAnalysis:
    """
    used to analyze the contribution of different paths.

    note that we kind of treat this class as immutable.
    all member functions return a new instance.
    """

    def __init__(self):
        self.source_list = []

    def add_source(self, *, conv, scale):
        ret = deepcopy(self)
        ret.source_list.append(
            {
                'conv': conv,
                'scale': scale,
            }
        )
        return ret

    def evaluate(self, scale_map=None, conv_map=None, check_all_used=False):
        # each scale is assigned some float number.
        # output a conv->float dict

        if scale_map is None:
            scale_map = dict()

        if conv_map is None:
            conv_map = dict()

        output = defaultdict(float)

        used_symbols = set()

        for x in self.source_list:
            conv, scale = x['conv'], x['scale']
            used_symbols.update({s for s in scale if type(s) is str})
            used_symbols.update({c for c in conv})
            scale_materialized = [scale_map[s] if type(s) is str else s for s in scale]
            conv_materialized = [conv_map[c] for c in conv]
            for z in scale_materialized:
                assert type(z) is float
            for c in conv_materialized:
                assert type(c) is float

            output[conv] += reduce((lambda x1, x2: x1 * x2), scale_materialized + conv_materialized, 1.0)
        if check_all_used:

            if used_symbols != (scale_map.keys() | conv_map.keys()):
                print(self.source_list)
                print(used_symbols, (scale_map.keys() | conv_map.keys()))
                print(scale_map, conv_map)
            assert used_symbols == (scale_map.keys() | conv_map.keys())

        return dict(output)

    def apply_scale(self, scale):
        ret = deepcopy(self)
        for x in ret.source_list:
            x['scale'] += scale
        return ret

    def apply_conv(self, conv):
        ret = deepcopy(self)
        for x in ret.source_list:
            x['conv'] += conv
        return ret

    def add(self, another):
        ret = deepcopy(self)
        ret.source_list.extend(deepcopy(another.source_list))
        return ret

    @property
    def unique_convs(self):
        ret = set()
        for x in self.source_list:
            ret.add(x['conv'])
        return ret


def get_average_source(source_list):
    num_source = len(source_list)
    source_list = [s.apply_scale((1.0 / num_source,)) for s in source_list]
    # add them together
    return reduce((lambda x1, x2: x1.add(x2)), source_list)

def verify(src: LayerSourceAnalysis):
    for chain in src.source_list:
        assert chain['scale'] == (1.0,)
        assert chain['conv'][0] == 'I'
        assert len(chain['conv']) % 2 == 1
        for idx in range(0, len(chain['conv']) - 1, 2):
            assert chain['conv'][1+idx][0] in {'B', 'R'}
            assert chain['conv'][1+idx+1][0] == 's'


def extract_l_and_t(s):
    assert s[0] == 's'
    layer, time = s[1:].split(',')
    return int(layer) - 1, int(time) - 1


def separate_bn_rewrite(term_tuple, counter):
    new_term_tuple = []
    for term in term_tuple:
        if type(term) is not str or term[0] != 's':
            new_term_tuple.append(term)
        else:
            layer, time = extract_l_and_t(term)
            current_ctr = counter[layer, time]
            if current_ctr == 0:
                # first one, no name change.
                new_term_tuple.append(term)
            else:
                # append counter
                new_term_tuple.append(term + f',{current_ctr}')
            counter[layer, time] += 1
    return tuple(new_term_tuple)

def get_source_analysis_for_one_model_spec_separate_bn(sources_list_last):
    assert type(sources_list_last) is list
    counter = Counter()
    for chain_list_this in sources_list_last:
        assert type(chain_list_this) is LayerSourceAnalysis
        for chain in chain_list_this.source_list:
            # scale related terms should appear in EXACTLY ONE OF `conv` and `scale`.
            # this is not formally checked, though
            chain['conv'] = separate_bn_rewrite(chain['conv'], counter)
            chain['scale'] = separate_bn_rewrite(chain['scale'], counter)
    return

def get_source_analysis_for_one_model_spec(*, num_recurrent_layer, num_cls, readout_type, return_raw=False,
                                           add_bn_in_chain=False, separate_bn=False):
    assert num_recurrent_layer >= 1
    assert num_cls >= 1

    if readout_type == 'legacy':
        assert num_cls == 1

    # first, get data at all iterations, inst
    # here, `I` is the feedforward input, which is constant across all time steps.

    # sources_list is a 2D list, indexed by [layer][time]
    # eventually it should have `num_recurrent_layer+1` x `num_cls` elements.
    # +1 because there is a feedforward layer.

    # for the first layer (feed forward), the output at each time step is the same.
    sources_list = [
        [LayerSourceAnalysis().add_source(conv=('I',), scale=(1.0,))]*num_cls
    ]

    # go over each recurrent layer
    for layer_idx in range(num_recurrent_layer):

        # get the names for scale (S), feedforward (Bottom up) and lateral (Recurrent) operators.
        layer_idx_human = layer_idx + 1
        sources_this = []
        conv_symbol_b = f'B{layer_idx_human}'
        conv_symbol_r = f'R{layer_idx_human}'
        scale_prefix = f's{layer_idx_human}'

        # go over each time step.
        for t in range(num_cls):
            # each iteration has a different scale operator.
            scale_this = f'{scale_prefix},{t+1}'
            # take previous layer's output at this time step.
            src_this = sources_list[-1][t]
            # apply B convolution
            src_this = src_this.apply_conv((conv_symbol_b,))
            if t > 0:
                # take last iteration's output and take R convolution, add the result
                src_this = src_this.add(sources_this[-1].apply_conv((conv_symbol_r,)))
            # BN.
            # this add_bn_in_chain is for purely symbolic computation.
            # to implement multi path version of RL block.
            if add_bn_in_chain:
                # it's a hack. in the original implementation of LayerSourceAnalysis,
                # it's not supposed to capture the exact order of conv and BN layers,
                # only the relative order among convs and order among BNs themselves.
                src_this = src_this.apply_conv((scale_this,))

                # check that it's always I + pairs of (B, s)
                verify(src_this)
            else:
                src_this = src_this.apply_scale((scale_this,))

            sources_this.append(src_this)
        sources_list.append(sources_this)

    if separate_bn:
        # this is needed so that I can evaluate the multi path models.
        # the last layer is rewritten.
        get_source_analysis_for_one_model_spec_separate_bn(sources_list[-1])

    if return_raw:
        return sources_list

    if readout_type in {'inst-last', 'legacy'}:
        return sources_list[-1][-1]
    elif readout_type in {'cm-last', 'inst-avg'}:
        # one averaging. my source analysis does not tell the difference between the two.
        return get_average_source(sources_list[-1])
    elif readout_type == 'cm-avg':
        sources = [get_average_source(sources_list[-1][:idx + 1]) for idx in range(num_cls)]
        return get_average_source(sources)
    else:
        raise ValueError
