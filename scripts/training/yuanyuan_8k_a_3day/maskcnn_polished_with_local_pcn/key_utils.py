from typing import Union


# put it here so that it can be used in submit as well.
def keygen(*,
           split_seed: Union[int, str],
           model_seed: int,
           act_fn: str,
           loss_type: str,
           input_size: int,
           out_channel: int,
           num_layer: int,
           kernel_size_l1: int,
           pooling_ksize: int,
           scale_name: str,
           smoothness_name=str,
           pooling_type: str,
           bn_before_act: bool,
           bn_after_fc: bool,

           pcn_cls: int,
           pcn_bypass: bool,
           pcn_no_act: bool,
           pcn_bn_post: bool,
           pcn_final_act: bool,
           pcn_bn: bool,
           pcn_bias: bool,
           ):
    # suffix itself can contain /
    return '/'.join([
        'yuanyuan_8k_a_3day/maskcnn_polished_with_local_pcn',
        f's_se{split_seed}',
        f'in_sz{input_size}',
        f'out_ch{out_channel}',
        f'num_l{num_layer}',
        f'k_l1{kernel_size_l1}',
        f'k_p{pooling_ksize}',
        f'pt{pooling_type}',
        f'bn_b_act{bn_before_act}',
        f'bn_a_fc{bn_after_fc}',
        f'act{act_fn}',

        # add those local pcn specific stuff
        f'p_c{pcn_cls}',
        f'p_bypass{pcn_bypass}',
        f'p_n_act{pcn_no_act}',
        f'p_bn_p{pcn_bn_post}',
        f'p_act{pcn_final_act}',
        f'p_bn{pcn_bn}',
        f'p_bias{pcn_bias}',

        f'sc{scale_name}',
        f'sm{smoothness_name}',
        f'l{loss_type}',
        f'm_se{model_seed}'
    ])


def script_keygen(**kwargs):
    # remove scale and smoothness
    del kwargs['scale']
    del kwargs['smoothness']
    key = keygen(**kwargs)

    # remove yuanyuan_8k_a_3day/maskcnn_polished part
    return '+'.join(key.split('/')[2:])
