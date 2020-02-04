"""(custom) building blocks for models"""
from typing import List

try:
    from torchnetjson.module import register_module_custom
    from torchnetjson.init import (
        register_init_custom, standard_init, bn_init_passthrough
    )
except ImportError:
    register_module_custom = None
    register_init_custom = None
    standard_init = None
    bn_init_passthrough = None

_store = dict()


def load_modules(mods: List[str]):
    # load all modules
    from . import maskcnn, localpcn_purdue, rcnn_basic_kriegeskorte

    for mod in mods:
        mod_info = _store[mod]
        register_module_custom(mod, mod_info['module'])
        init_info = mod_info['init']
        if init_info is not None:
            register_init_custom(mod, init_info)


def register_module(name, module, init=None):
    assert name not in _store
    _store[name] = {'module': module, 'init': init}
