from .. import register_module, standard_init

from .reference_rewrite import PcConvBp


# pcn local init
def pcconvbp_init(mod: PcConvBp, init: dict) -> None:
    standard_init(mod, init, attrs_to_init=('FFconv.weight',
                                            'FBconv.weight',
                                            'bypass.weight'),
                  strict=False)


register_module('localpcn_purdue.baseline', PcConvBp, pcconvbp_init)
