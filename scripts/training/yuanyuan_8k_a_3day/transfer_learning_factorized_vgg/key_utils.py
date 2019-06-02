

# put it here so that it can be used in submit as well.
def keygen(*,
           split_seed: int,
           sparse: str,
           model_seed: int,
           act_fn: str,
           loss_type: str,
           suffix: str,
           ):
    # suffix itself can contain /
    suffix = suffix.replace('/', '=')
    return f'yuanyuan_8k_a_3day/transfer_learning_factorized_vgg/split_seed{split_seed}/act{act_fn}/sparse{sparse}/{suffix}/loss{loss_type}/model_seed{model_seed}'  # noqa: E501


def script_keygen(*,
                  split_seed: int,
                  sparse: str,
                  model_seed: int,
                  act_fn: str,
                  loss_type: str,
                  suffix: str,
                  ):
    key = keygen(split_seed=split_seed,
                 sparse=sparse,
                 model_seed=model_seed,
                 act_fn=act_fn,
                 loss_type=loss_type,
                 suffix=suffix)

    # remove yuanyuan_8k_a_3day/transfer_learning_factorized_vgg part
    return '+'.join(key.split('/')[2:])
