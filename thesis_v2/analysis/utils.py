from collections import defaultdict
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

    def evaluate(self, scale_map=None, conv_map=None):
        # each scale is assigned some float number.
        # output a conv->float dict

        if scale_map is None:
            scale_map = dict()

        if conv_map is None:
            conv_map = dict()

        output = defaultdict(float)
        for x in self.source_list:
            conv, scale = x['conv'], x['scale']
            scale_materialized = [scale_map.get(s, s) for s in scale]
            conv_materialized = [conv_map.get(c, 1.0) for c in conv]
            for z in scale_materialized:
                assert type(z) is float
            for c in conv_materialized:
                assert type(c) is float

            output[conv] += reduce((lambda x1, x2: x1 * x2), scale_materialized + conv_materialized, 1.0)
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


def get_source_analysis_for_one_model_spec(*, num_recurrent_layer, num_cls, readout_type, return_raw=False):
    assert num_recurrent_layer == 1
    assert num_cls >= 1
    # first, get data at all iterations, inst

    sources = [LayerSourceAnalysis().add_source(conv=('B',), scale=('s1',))]

    for t in range(2, num_cls + 1):
        src_this = LayerSourceAnalysis().add_source(
            conv=('B',), scale=()
        )
        # input from recurrent
        src_prev_with_recurrent = sources[-1].apply_conv(('R',))

        src_this = src_this.add(src_prev_with_recurrent)
        src_this = src_this.apply_scale((f's{t}',))

        sources.append(src_this)

    assert len(sources) == num_cls

    if return_raw:
        return sources

    if readout_type == 'inst-last':
        return sources[-1]
    elif readout_type in {'cm-last', 'inst-avg'}:
        # one averaging. my source analysis does not tell the difference between the two.
        return get_average_source(sources)
    elif readout_type == 'cm-avg':
        sources = [get_average_source(sources[:idx + 1]) for idx in range(num_cls)]
        return get_average_source(sources)
    else:
        raise ValueError
