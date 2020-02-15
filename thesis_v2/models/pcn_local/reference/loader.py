"""this legacy model is not to be built using JSON builder,
instead, we only load some pretrained weights.

the pretrained model is assumed to trained using fp16 trick,

following the setup in https://github.com/leelabcnbc/thesis-yimeng-v1/blob/25471e6e80f7acd0f2ec82bb9c577c58bfdd7171/3rdparty/PCN-with-Local-Recurrent-Processing/main_imagenet_fp16.py

"""  # noqa: E501
from os.path import join
import torch.optim
from torch import nn

from .... import dir_dict

# apex_fp16_utils.LossScaler is still needed because that checkpoint has an LossScaler
# inside.
# from apex_fp16_utils import LossScaler

from leelabtoolbox.feature_extraction.cnn.generic_network_definitions import (
    _create_blob_info_dict
)

from leelabtoolbox.feature_extraction.cnn import cnnsizehelper

from .prednet import PredNetBpE


def get_pretrained_network(net_name,
                           root_dir=join(
                               dir_dict['root'], '3rdparty', 'PCN-with-Local-Recurrent-Processing', 'checkpoint'
                           )):
    if net_name == 'PredNetBpE_3CLS':
        best_file = join(root_dir,
                         'model_best.pth.tar.3CLS')
        trained_model = load_pcn_imagenet('PredNetBpE', 3,
                                          checkpoint_path=best_file)[0]
    elif net_name == 'PredNetBpE_1CLS':
        best_file = join(root_dir,
                         'model_best.pth.tar.1CLS')
        trained_model = load_pcn_imagenet('PredNetBpE', 1,
                                          checkpoint_path=best_file)[0]
    elif net_name == 'PredNetBpE_0CLS':
        best_file = join(root_dir,
                         'model_best.pth.tar.0CLS')
        trained_model = load_pcn_imagenet('PredNetBpE', 0,
                                          checkpoint_path=best_file)[0]
    else:
        raise ValueError

    return trained_model


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = network

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


def load_pcn_imagenet(model_name, num_cycle, num_class=1000,
                      *, checkpoint_path):
    models = {'PredNetBpE': PredNetBpE}
    print("=> creating model '{}'".format(
        model_name + '_' + str(num_cycle) + 'CLS')
    )
    model = models[model_name](num_classes=num_class, cls=num_cycle)
    model = torch.nn.DataParallel(model)
    model = FP16Model(model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # get out the parallel part.
    model = model.network.module

    # because DataParallel by default can put things through GPU.
    model.cpu()

    # convert back to float32
    # model.float()
    model.eval()

    # you don't have to transfer FP32 master parameter to the network,
    # as the network was not tested that way anyway,
    # and those precision loss should be tolerable.

    # return a CPU, eval-ready model.
    return model, checkpoint


def get_layer_info_dict(model_name):
    assert model_name == 'PredNetBpE'
    return _create_blob_info_dict([
        ('convb', 2, 7, 3),
        ('conv0', 1, 3, 1),
        ('conv1', 1, 3, 1),
        ('conv2', 1, 3, 1),
        ('pool2', 2, 2, 0),
        ('conv3', 1, 3, 1),
        ('conv4', 1, 3, 1),
        ('pool4', 2, 2, 0),
        ('conv5', 1, 3, 1),
        ('conv6', 1, 3, 1),
        ('pool6', 2, 2, 0),
        ('conv7', 1, 3, 1),
        ('conv8', 1, 3, 1),
        ('conv9', 1, 3, 1),
        ('pool9', 2, 2, 0),
        ('conv10', 1, 3, 1),
        ('poolfc', 7, 7, 0),
    ])


def get_layers_to_extract(model_name, order='legacy'):
    assert model_name == 'PredNetBpE'
    # you should infer this from get_layer_info_dict.
    # expanding every convX into convX.in, convX.init, convX.loop
    list_init = get_layer_info_dict(model_name).keys()

    list_new = []

    for layer_this in list_init:
        if order == 'legacy':
            list_new.append(layer_this)
            if layer_this.startswith('conv') and layer_this != 'convb':
                # then we expand
                list_new.append(layer_this + '.in')
                list_new.append(layer_this + '.init')
                list_new.append(layer_this + '.loop')
        elif order == 'computation':  # in order of computation,
            if layer_this.startswith('conv') and layer_this != 'convb':
                # then we expand
                list_new.append(layer_this + '.in')
                list_new.append(layer_this + '.init')
                list_new.append(layer_this + '.loop')
            list_new.append(layer_this)
        else:
            raise NotImplementedError

    return list_new


def get_layer_residuals(model_name, num_loop):
    assert isinstance(num_loop, int)
    # loop - init, loop-loop, and final output - last loop for num_loop!=0
    # final output - init for num_loop=0
    layers_to_extract = get_layers_to_extract(model_name, order='computation')
    # basically, search everything ending in .loop, and add that one minus
    # previous one, and next one minus this one.

    idx_loops = [idx for idx, n in enumerate(layers_to_extract) if
                 n.endswith('.loop')]

    result = []
    for idx in idx_loops:
        if num_loop != 0:
            # loop0 - init
            result.append(
                (
                    (layers_to_extract[idx - 1], 0),
                    (layers_to_extract[idx], 0),
                )
            )
            # loop - loop
            # output - last loop

            for idx_loop in range(num_loop - 1):
                result.append(
                    (
                        (layers_to_extract[idx], idx_loop),
                        (layers_to_extract[idx], idx_loop + 1),
                    )
                )

            result.append(
                (
                    (layers_to_extract[idx], num_loop - 1),
                    (layers_to_extract[idx + 1], 0),
                )
            )
        else:
            result.append(
                # (layer_name0, unroll_idx0),(layer_name1, unroll_idx1)
                (
                    (layers_to_extract[idx - 1], 0),
                    (layers_to_extract[idx + 1], 0)
                )
            )
    return result


def get_layer_residuals_numerical(model_name, num_loop, prefix='',
                                  order='legacy', concat_symbol='#'):
    # for generating submission scripts.
    names = get_layer_residuals(model_name, num_loop)
    additional_cases = []

    layernames = get_layers_to_extract(model_name, order)

    for (layername0, idx0), (layername1, idx1) in names:
        suffix_this = concat_symbol.join(
            [
                join(prefix, f'{layernames.index(layername0)}.{idx0}'),
                f'{layernames.index(layername1)}.{idx1}'
            ]
        )
        additional_cases.append(suffix_this)

    return additional_cases


def slice_dict_name_mapping(model_name):
    assert model_name == 'PredNetBpE'
    # you should infer this from get_layer_info_dict.
    # expanding every convX into convX.in, convX.init, convX.loop
    list_init = get_layer_info_dict(model_name).keys()

    mapping = dict()
    prev_layer = None
    for layer_this in list_init:
        if layer_this.startswith('conv') and layer_this != 'convb':
            # then we expand
            mapping[layer_this] = layer_this
            mapping[layer_this + '.loop'] = layer_this
            mapping[layer_this + '.init'] = layer_this
            assert prev_layer is not None
            mapping[layer_this + '.in'] = prev_layer
        else:
            mapping[layer_this] = layer_this
        prev_layer = layer_this

    return mapping


def get_name_mapping(model_name):
    assert model_name == 'PredNetBpE'

    mapping = dict()

    for layer in get_layer_info_dict(model_name).keys():
        # order of these matter.
        if layer == 'convb':
            keys = ('baseconv',)
            values = (layer,)
        elif layer == 'poolfc':
            keys = ('fc',)
            values = (layer,)
        elif layer.startswith('pool'):
            keys = (f'maxpool_dict.{layer[4:]}',)
            values = (layer,)
        elif layer.startswith('conv'):
            layer_idx = layer[4:]
            keys = (
                f'PcConvs.{layer_idx}',
                f'BNs.{layer_idx}',
                f'PcConvs.{layer_idx}.resp_init',
                f'PcConvs.{layer_idx}.resp_loop',
            )
            values = (
                layer,
                layer + '.in',  # input of this layer, which has same
                # RF of previous layer.
                layer + '.init',
                layer + '.loop',
            )
        else:
            raise ValueError

        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            assert k not in mapping
            mapping[k] = v

    return mapping


def get_slicing_dict(model_name, rf_size):
    assert model_name == 'PredNetBpE'
    layer_info = get_layer_info_dict(model_name)

    input_size = (224, 224)

    helper = cnnsizehelper.CNNSizeHelper(layer_info, input_size, False)

    slicing_dict = dict()
    fc_layers = {'poolfc'}

    for layer in helper.layer_info_dict:
        if layer not in fc_layers:
            if rf_size is not None:
                top_bottom = input_size[0] / 2 - rf_size / 2, input_size[
                    0] / 2 + rf_size / 2
                left_right = input_size[1] / 2 - rf_size / 2, input_size[
                    1] / 2 + rf_size / 2
                slicing_dict[layer] = helper.compute_minimum_coverage(layer,
                                                                      top_bottom,
                                                                      left_right)
            else:
                slicing_dict[layer] = ((None, None), (None, None))

    slicing_dict = cnnsizehelper.get_slice_dict(slicing_dict,
                                                slicing_dict.keys())
    for fc_layer in fc_layers:
        slicing_dict[fc_layer] = ()
    # then map using slice_dict_name_mapping

    slicing_dict_actual = dict()
    for k, v in slice_dict_name_mapping(model_name).items():
        slicing_dict_actual[k] = slicing_dict[v]
    return slicing_dict_actual
