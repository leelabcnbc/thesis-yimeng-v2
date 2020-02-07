"""generates the config file needed for a mask CNN"""

from collections import OrderedDict

# try to import torchnetjson and register
from ...blocks import load_modules as load_modules_global

from ...blocks_json import general, utils, maskcnn, pooling, rcnn_kriegeskorte


def gen_maskcnn_polished_with_rcnn_k_bl(
        input_size, num_neuron, *,
        out_channel=48,
        kernel_size_l1=13, kernel_size_l23=3,
        factored_constraint=None,
        act_fn='softplus',
        do_init=True,
        pooling_type='max',
        pooling_ksize=2,
        num_layer=3,
        bn_after_fc=False,
        # 1 means feedforward
        n_timesteps=1,
        blstack_pool_ksize=1,
        acc_mode='cummean',
        debug_args=None,
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

    utils.update_module_dict(module_dict,
                             rcnn_kriegeskorte.blstack(
                                 name='bl_stack',
                                 input_size=input_size_dict['map_size'],
                                 n_timesteps=n_timesteps,
                                 channel_list=[1, ] + [out_channel, ] * num_layer,
                                 kernel_size_list=[kernel_size_l1, ] + [kernel_size_l23, ] * (num_layer - 1),
                                 pool_ksize=blstack_pool_ksize,
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
    conv_stage_modules = {'bn_input', 'bl_stack', 'accumulator'}
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
            'conv_layers': [f'stack.layer_list.{x}.b_conv' for x in range(num_layer)]
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
