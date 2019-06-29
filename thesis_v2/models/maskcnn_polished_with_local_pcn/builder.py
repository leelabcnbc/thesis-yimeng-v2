"""generates the config file needed for a mask CNN"""

from collections import OrderedDict

# try to import torchnetjson and register
from ...blocks import load_modules as load_modules_global

from ...blocks_json import general, conv, utils, maskcnn, pooling, local_pcn_purdue


def gen_maskcnn_polished_with_local_pcn(
        input_size, num_neuron, *,
        out_channel=48,
        kernel_size_l1=13, kernel_size_l23=3,
        factored_constraint=None,
        act_fn='softplus',
        do_init=True,
        pooling_type='max',
        pooling_ksize=2,
        num_layer=3,
        bn_before_act=True,
        bn_after_fc=False,

        # local pcn specific stuff.
        pcn_cls: int,
        pcn_bypass: bool,
        pcn_tied: bool,
        pcn_no_act: bool,
        pcn_b0_init: float,
        pcn_bn_post: bool,
        pcn_final_act: bool,
        pcn_bn: bool,
        pcn_bias: bool,


        bn_locations_legacy=False,
        # if True,
        # bn -> conv -> bn_post -> act
        # to match what's used before.

        # if False,
        # it's conv -> bn -> act -> bn_post.
        # this may interface better with maskcnn_polished.
):
    assert num_layer >= 1
    assert kernel_size_l23 % 2 == 1

    input_size = utils.check_input_size(input_size)

    module_dict = OrderedDict()

    utils.update_module_dict(module_dict,
                             general.bn(
                                 name='bn_input',
                                 num_features=1,
                                 affine=True, do_init=do_init,
                             )
                             )

    input_size_dict = {'map_size': input_size}
    del input_size

    utils.update_module_dict(module_dict,
                             conv.conv2dstack(
                                 input_size=input_size_dict['map_size'],
                                 suffix='0',
                                 kernel_size=kernel_size_l1,
                                 in_channel=1,
                                 out_channel=out_channel,
                                 act_fn=act_fn,
                                 bn_before_act=bn_before_act,
                                 state_dict=input_size_dict,
                             ))

    for layer_idx in range(1, num_layer):
        utils.update_module_dict(module_dict,
                                 local_pcn_purdue.conv2d_baseline(
                                     input_size=input_size_dict['map_size'],
                                     suffix=str(layer_idx),
                                     kernel_size=kernel_size_l23,
                                     in_channel=out_channel,
                                     out_channel=out_channel,
                                     act_fn=act_fn,
                                     state_dict=input_size_dict,
                                     padding=kernel_size_l23 // 2,
                                     localpcn_bypass=pcn_bypass,
                                     localpcn_b0_init=pcn_b0_init,
                                     localpcn_cls=pcn_cls,
                                     localpcn_tied=pcn_tied,
                                     final_act=pcn_final_act,
                                     bn_post=pcn_bn_post,
                                     localpcn_no_act=pcn_no_act,
                                     bn=pcn_bn,
                                     bn_locations_legacy=bn_locations_legacy,
                                     localpcn_bias=pcn_bias,
                                 ))

    # print(input_size)
    # a max pooling to reduce number of parameters to about 1/4.

    utils.update_module_dict(module_dict,
                             pooling.pool2d(
                                 name='pooling',
                                 pooling_type=pooling_type,
                                 kernel_size=pooling_ksize,
                                 input_size=input_size_dict['map_size'],
                                 state_dict=input_size_dict,
                                 ceil_mode=True,
                                 map_size_strict=False,
                             ))

    # factored fc
    constraint_this = factored_constraint
    if not isinstance(constraint_this, tuple):
        constraint_this = (constraint_this, constraint_this)
    assert isinstance(constraint_this, tuple) and len(constraint_this) == 2
    utils.update_module_dict(
        module_dict,
        maskcnn.factoredfc(
            name='fc',
            map_size=input_size_dict['map_size'],
            in_channels=out_channel,
            out_features=num_neuron,
            bias=not bn_after_fc,
            weight_spatial_constraint=constraint_this[0],
            weight_feature_constraint=constraint_this[1],
        )
    )

    # this bn_after_fc stuff is poor in previous experiments.
    # here I add it back, just for confirmation.

    if bn_after_fc:
        utils.update_module_dict(
            module_dict,
            general.bn(
                name='bn_output',
                num_features=num_neuron,
                do_init=do_init,
                affine=True,
                ndim=1,
            )
        )
    # I cannot put bn after act,
    # as I use poisson loss which requires non negative neural response,
    # and poisson loss overall performs a bit better than MSE loss
    # (check <https://github.com/leelabcnbc/thesis-yimeng-v2/blob/2f49f91e70eb48a1ab1d6917374fa40ee14ef50c/results_processed/yuanyuan_8k_a_3day/transfer_learning_factorized_vgg/vgg.ipynb>  # noqa: E501
    # and poisson loss makes more sense for neural data

    utils.update_module_dict(
        module_dict,
        general.act(name='final_act',
                    act_fn=act_fn)
    )

    fc_stage_modules = {'pooling', 'fc', 'bn_output', 'final_act'}

    param_dict = utils.generate_param_dict(
        module_dict=module_dict,
        op_params=[
            {
                'type': 'sequential',
                'param': (lambda idx, x: x not in fc_stage_modules),
                'in': 'input0',
                'out': 'feature_map',
                'keep_out': False,
            },
            {
                'type': 'sequential',
                'param': (lambda idx, x: x in fc_stage_modules),
                'in': 'feature_map',
                'out': 'out_neural',
                'keep_out': True,
            }
        ],
        comments={
            'conv_layers': [
                               'conv0',
                           ] + [
                               # FF conv in PCN blocks
                               f'conv{x}.FFconv' for x in range(1,
                                                                num_layer)
                           ],
        }
    )
    return param_dict


# register
def load_modules():
    load_modules_global([
        'maskcnn.factoredfc',
        'localpcn_purdue.baseline',
    ])
