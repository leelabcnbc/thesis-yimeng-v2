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
           ):
    # suffix itself can contain /
    return '/'.join([
        'yuanyuan_8k_a_3day/maskcnn_polished',
        f'split_seed{split_seed}',
        f'in_sz{input_size}',
        f'out_ch{out_channel}',
        f'num_l{num_layer}',
        f'ksize_l1{kernel_size_l1}',
        f'ksize_p{pooling_ksize}',
        f'ptype{pooling_type}',
        f'bn_before_act{bn_before_act}',
        f'bn_after_fc{bn_after_fc}',
        f'act{act_fn}',
        f'scale{scale_name}',
        f'smoothness{smoothness_name}',
        f'loss{loss_type}',
        f'model_seed{model_seed}'
    ])


def script_keygen(**kwargs):
    # remove scale and smoothness
    del kwargs['scale']
    del kwargs['smoothness']
    key = keygen(**kwargs)

    # remove yuanyuan_8k_a_3day/maskcnn_polished part
    return '+'.join(key.split('/')[2:])
