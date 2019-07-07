import numpy as np

from thesis_v2.data.prepared.crcns_pvc8 import natural_data

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v2.training_extra.maskcnn_like.training import (train_one,
                                                            partial)

from thesis_v2.models.maskcnn_polished.builder import (gen_maskcnn_polished, load_modules)

from key_utils import keygen

load_modules()


def master(*,
           shuffle_type,
           split_seed,
           model_seed,
           act_fn,
           loss_type,
           out_channel,
           num_layer,
           kernel_size_l1,
           pooling_ksize,
           scale, scale_name,
           smoothness, smoothness_name,
           pooling_type,
           bn_before_act,
           bn_after_fc,
           ):
    key = keygen(
        shuffle_type=shuffle_type,
        split_seed=split_seed,
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        out_channel=out_channel,
        num_layer=num_layer,
        kernel_size_l1=kernel_size_l1,
        pooling_ksize=pooling_ksize,
        scale_name=scale_name,
        smoothness_name=smoothness_name,
        pooling_type=pooling_type,
        bn_before_act=bn_before_act,
        bn_after_fc=bn_after_fc,
    )

    # keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
    datasets = natural_data('large', 144, 4, split_seed, scale=1 / 50, shuffle_type=shuffle_type)

    datasets = {
        'X_train': datasets[0].astype(np.float32),
        'y_train': datasets[1],
        'X_val': datasets[2].astype(np.float32),
        'y_val': datasets[3],
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }

    print(
        'train', datasets['X_train'].shape[0],
        'val', datasets['X_val'].shape[0],
        'test', datasets['X_test'].shape[0],
    )

    def gen_cnn_partial(input_size_cnn, n):
        return gen_maskcnn_polished(input_size=input_size_cnn,
                                    num_neuron=n,
                                    out_channel=out_channel,
                                    kernel_size_l1=kernel_size_l1,  # (try 5,9,13)
                                    kernel_size_l23=3,
                                    act_fn=act_fn,
                                    pooling_ksize=pooling_ksize,  # (try, 1,3,5,7)
                                    pooling_type=pooling_type,  # try (avg, max)  # looks that max works well here?
                                    num_layer=num_layer,
                                    bn_before_act=bn_before_act,
                                    bn_after_fc=bn_after_fc,
                                    )

    opt_config_partial = partial(get_maskcnn_v1_opt_config,
                                 scale=scale,
                                 smoothness=smoothness,
                                 group=0.0,
                                 loss_type=loss_type,
                                 )

    result = train_one(
        arch_json_partial=gen_cnn_partial,
        opt_config_partial=opt_config_partial,
        datasets=datasets,
        key=key,
        show_every=1000,
        max_epoch=40000,
        model_seed=model_seed,
        return_model=False
    )

    print(result['stats_best']['stats']['test']['corr_mean'])
