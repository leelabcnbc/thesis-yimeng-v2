"""similar to pcn_local.reference.loader,
to handle vgg network extraction for all types (11,13,16,19, etc.)

previously, I did this in a quite manual way.

hopefully this times code can be more compact.
"""
from leelabtoolbox.feature_extraction.cnn import cnnsizehelper
# noinspection PyProtectedMember
#   I just don't want to change leelab-toolbox for now.
from leelabtoolbox.feature_extraction.cnn.generic_network_definitions import (
    _create_blob_info_dict
)

# from <https://github.com/pytorch/vision/blob/v0.2.1/torchvision/models/vgg.py#L78-L83>  # noqa: E501

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,
          'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_mapping = {
    'vgg11': 'A',
    'vgg13': 'B',
    'vgg16': 'D',
    'vgg19': 'E',
    'vgg11_bn': 'A',
    'vgg13_bn': 'B',
    'vgg16_bn': 'D',
    'vgg19_bn': 'E',
}


def get_layer_info_dict(model_name):
    # replaces <https://github.com/leelabcnbc/leelab-toolbox/blob/e00376cb09e8ac535b4edbd65db20745fed0de54/leelabtoolbox/feature_extraction/cnn/generic_network_definitions.py#L46-L90>  # noqa: E501
    # follow standard naming
    idx_major = 1
    idx_minor = 1
    layer_info_raw = []
    for c in cfg[cfg_mapping[model_name]]:
        if isinstance(c, int):
            # add a new 3x3 conv layer
            layer_info_raw.append((f'conv{idx_major}_{idx_minor}',
                                   1, 3, 1))
            idx_minor += 1
        elif c == 'M':
            layer_info_raw.append((f'pool{idx_major}',
                                   2, 2, 0))
            idx_major += 1
            idx_minor = 1
        else:
            raise ValueError
    return _create_blob_info_dict(layer_info_raw)


def get_layers_to_extract(model_name):
    # all conv/pool layers + 2 fc layers.
    return list(get_layer_info_dict(model_name).keys()) + ['fc6', 'fc7']


def get_name_mapping(model_name):
    bn = model_name.endswith('_bn')

    result = dict()

    # separate cases for bn and no bn.
    layer_last = -1
    idx_major = 1
    idx_minor = 1
    for c in cfg[cfg_mapping[model_name]]:
        if isinstance(c, int):
            # add a new 3x3 conv layer
            if bn:
                layer_last += 3
            else:
                layer_last += 2

            result[f'features.{layer_last}'] = f'conv{idx_major}_{idx_minor}'
            idx_minor += 1
        elif c == 'M':
            layer_last += 1
            result[f'features.{layer_last}'] = f'pool{idx_major}'
            idx_major += 1
            idx_minor = 1
        else:
            raise ValueError

    result['classifier.1'] = 'fc6'
    result['classifier.4'] = 'fc7'

    return result


def get_slicing_dict(
        *, model_name, rf_size, input_size=(224, 224)
):
    layer_info = get_layer_info_dict(model_name)

    top_bottom = (input_size[0] / 2 - rf_size / 2,
                  input_size[0] / 2 + rf_size / 2)
    left_right = (input_size[1] / 2 - rf_size / 2,
                  input_size[1] / 2 + rf_size / 2)

    helper = cnnsizehelper.CNNSizeHelper(layer_info, input_size, True)

    slicing_dict = dict()
    for layer in helper.layer_info_dict:
        slicing_dict[layer] = helper.compute_minimum_coverage(layer,
                                                              top_bottom,
                                                              left_right)
    slicing_dict = cnnsizehelper.get_slice_dict(slicing_dict,
                                                slicing_dict.keys())
    for fc_layer in {'fc6', 'fc7'}:
        slicing_dict[fc_layer] = ()
    # then map using slice_dict_name_mapping

    return slicing_dict
