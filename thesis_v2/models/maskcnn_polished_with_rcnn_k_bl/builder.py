"""generates the config file needed for a mask CNN"""

from collections import OrderedDict

# try to import torchnetjson and register
from ...blocks import load_modules as load_modules_global

from ...blocks_json import general, utils, maskcnn, pooling, rcnn_kriegeskorte, conv


def gen_maskcnn_polished_with_rcnn_k_bl(
        input_size, num_neuron, *,
        # = 48
        out_channel,
        # = 13
        kernel_size_l1,
        # = 3
        kernel_size_l23,
        # = None,
        factored_constraint,
        # = 'softplus'
        act_fn,
        # ='max'
        pooling_type,
        # =2
        pooling_ksize,
        # =3,
        num_layer,
        # =1 means feedforward
        n_timesteps,
        # =1,
        blstack_pool_ksize,
        # =None,
        blstack_pool_type,
        # ='cummean',
        acc_mode,
        # =False,
        bn_after_fc,
        do_init=True,
        debug_args=None,
        # first conv block being purely feedforward, matching the behavior in maskcnn_polished_with_local_pcn
        ff_1st_block=False,
        ff_1st_bn_before_act=True,
):
    assert num_layer >= 1
    assert kernel_size_l1 % 2 == 1
    assert kernel_size_l23 % 2 == 1

    input_size = utils.check_input_size(input_size)

    if debug_args is None:
        debug_args = dict()

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

    if not ff_1st_block:
        utils.update_module_dict(module_dict,
                                 rcnn_kriegeskorte.blstack(
                                     name='bl_stack',
                                     input_size=input_size_dict['map_size'],
                                     n_timesteps=n_timesteps,
                                     channel_list=[1, ] + [out_channel, ] * num_layer,
                                     kernel_size_list=[kernel_size_l1, ] + [kernel_size_l23, ] * (num_layer - 1),
                                     pool_ksize=blstack_pool_ksize,
                                     pool_type=blstack_pool_type,
                                     act_fn=act_fn,
                                     do_init=do_init,
                                     state_dict=input_size_dict,
                                     )
                                )
    else:
        # copied from thesis_v2.models.maskcnn_polished_with_local_pcn.builder.gen_maskcnn_polished_with_local_pcn
        utils.update_module_dict(module_dict,
                                 conv.conv2dstack(
                                     input_size=input_size_dict['map_size'],
                                     suffix='0',
                                     kernel_size=kernel_size_l1,
                                     in_channel=1,
                                     out_channel=out_channel,
                                     act_fn=act_fn,
                                     bn_before_act=ff_1st_bn_before_act,
                                     state_dict=input_size_dict,
                                 ))
        utils.update_module_dict(module_dict,
                                 rcnn_kriegeskorte.blstack(
                                     name='bl_stack',
                                     input_size=input_size_dict['map_size'],
                                     n_timesteps=n_timesteps,
                                     channel_list=[out_channel, ] * num_layer,
                                     kernel_size_list=[kernel_size_l23, ] * (num_layer - 1),
                                     pool_ksize=blstack_pool_ksize,
                                     pool_type=blstack_pool_type,
                                     act_fn=act_fn,
                                     do_init=do_init,
                                     state_dict=input_size_dict,
                                 )
                                 )

    utils.update_module_dict(module_dict,
                             rcnn_kriegeskorte.accumulator(
                                 name='accumulator',
                                 mode=acc_mode,
                             )
                             )

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

    if not ff_1st_block:
        conv_stage_modules = {'bn_input', 'bl_stack', 'accumulator'}
        conv_layers_comment = [f'bl_stack.layer_list.{x}.b_conv' for x in range(num_layer)]
    else:
        conv_stage_modules = {'bn_input', 'conv0', 'bl_stack', 'accumulator'}
        conv_layers_comment = ['conv0'] + [f'bl_stack.layer_list.{x}.b_conv' for x in range(num_layer-1)]

    if debug_args.get('only_fc', False):
        fc_stage_modules = {'fc'}
    else:
        # this is for debugging. note that please set pooling_ksize to 1 to facilitate debugging.
        fc_stage_modules = {'pooling', 'fc', 'bn_output', 'final_act'}
    print(fc_stage_modules)

    use_stack = debug_args.get('stack', True)

    param_dict = utils.generate_param_dict(
        module_dict=module_dict,
        op_params=[
            {
                'type': 'sequential',
                'param': ((lambda idx, x: x in conv_stage_modules), {'module_op_kwargs': {'unpack': False}}),
                'in': 'input0',
                'out': 'feature_map',
                'keep_out': False,
            },
            {
                'type': 'sequential',
                'param': ((lambda idx, x: x in fc_stage_modules), {'module_op_name': 'module_repeat'}),
                'in': 'feature_map',
                'out': 'out_neural_separate',
                'keep_out': not use_stack,
            },
            # finally combine them together.

        ] + ([
            {
                'type': 'stack',
                'param': {'dim': 0},
                'in': 'out_neural_separate',
                'out': 'out_neural',
                'keep_out': True,
            }
        ] if use_stack else []),
        comments={
            'conv_layers': conv_layers_comment
        },
        output_list=use_stack
    )
    return param_dict


# register
def load_modules():
    load_modules_global([
        'maskcnn.factoredfc',
        'rcnn_kriegeskorte.blstack',
        'rcnn_kriegeskorte.accumulator',
    ])
