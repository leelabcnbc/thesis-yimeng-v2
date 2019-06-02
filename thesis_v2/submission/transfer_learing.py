# utils for transfer learning.

import json
import h5py


def fetch_meta(name: str, obj: h5py.Dataset):
    splitted_name = name.split('/')
    assert len(splitted_name) == 4
    dataset = splitted_name[0] == 'a'
    network = splitted_name[1]
    layer_idx, layer_unroll = splitted_name[3].split('.')
    layer_idx = int(layer_idx)
    layer_unroll = int(layer_unroll)

    config = json.loads(obj.parent.attrs['config'].decode('utf-8'))
    layer_name = config['module_names'][layer_idx]

    return {
        'dataset': dataset,
        'network': network,
        'layer_idx': layer_idx,
        'layer_name': layer_name,
        'layer_unroll': layer_unroll,
        'setting': splitted_name[2],
        'splitted_name': tuple(splitted_name),
    }
