"""generates the config file needed for a mask CNN"""

from typing import List
from collections import OrderedDict
from ...blocks_json import general, maskcnn
from ...blocks_json.utils import update_module_dict, generate_param_dict
from ...blocks import load_modules as load_modules_global


# setting affine to False follows what is written
# in paper
# [biorxiv V1 fitting paper "Deep convolutional models improve predictions of macaque V1 responses to natural images"](https://doi.org/10.1101/201764)  # noqa: E501
# (not sure exactly what they did)
#
# but at least this makes input scale stay stable and lambdas on regularization
# factors can be used across many networks.

def gen_transfer_learner(in_shape: List[int], num_neuron: int,
                         act_fn='softplus',
                         batchnorm=True,
                         batchnorm_affine=False,
                         do_init=True,
                         factorized=False,
                         ):
    # here I will assume that input is 4D (N, C, H, W),
    # and input_shape is (C, H, W).
    # this simplifies the design of optimizer.

    assert len(in_shape) == 3

    module_dict = OrderedDict()

    if batchnorm:
        # add batch norm
        update_module_dict(module_dict, general.bn(
            name='bn',
            num_features=in_shape[0],
            affine=batchnorm_affine,
            do_init=do_init,
        ))

    if not factorized:
        # this is not useful as the factorized one,
        # as shown in previous experiments.
        raise NotImplementedError
    else:
        # not tested for others. but should be ok.
        assert in_shape[0] > 1
        update_module_dict(
            module_dict,
            maskcnn.factoredfc(
                name='fc',
                map_size=(in_shape[1], in_shape[2]),
                out_features=num_neuron,
                in_channels=in_shape[0],
                do_init=do_init,
            ),
        )

    update_module_dict(module_dict,
                       general.act(name='final_act',
                                   act_fn=act_fn))
    # op_params=None is the default one.
    # using all modules, one input, one output.
    # simplest case.
    return generate_param_dict(module_dict=module_dict,
                               op_params=None)


# register
def load_modules():
    load_modules_global([
        'maskcnn.factoredfc',
    ])
