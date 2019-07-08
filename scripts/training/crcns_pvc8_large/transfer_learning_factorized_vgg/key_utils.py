from typing import Union


# put it here so that it can be used in submit as well.
def keygen(*,
           shuffle_type: str,
           split_seed: Union[int, str],
           sparse: str,
           model_seed: int,
           act_fn: str,
           loss_type: str,
           suffix: str,
           ):
    # suffix itself can contain /
    suffix = suffix.replace('/', '=')
    return f'crcns_pvc8_large/transfer_learning_factorized_vgg/shuffle_type{shuffle_type}/split_seed{split_seed}/act{act_fn}/sparse{sparse}/{suffix}/loss{loss_type}/model_seed{model_seed}'  # noqa: E501


def script_keygen(**kwargs):
    key = keygen(**kwargs)

    # remove crcns_pvc8_large/transfer_learning_factorized_vgg part
    return '+'.join(key.split('/')[2:])
