from typing import Union


# put it here so that it can be used in submit as well.
def keygen(*,
           shuffle_type: str,
           split_seed: Union[int, str],
           model_seed: int,
           act_fn: str,
           loss_type: str,
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
        'crcns_pvc8_large/maskcnn_polished',
        f'shuffle_type{shuffle_type}',
        f'split_seed{split_seed}',
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

    # remove 'crcns_pvc8_large/maskcnn_polished' part
    return '+'.join(key.split('/')[2:])
