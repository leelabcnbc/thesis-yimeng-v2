import numpy as np

from thesis_v2.data.prepared.yuanyuan_8k import get_data

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v2.training_extra.maskcnn_like.training import (train_one,
                                                            partial)

from thesis_v2.models.maskcnn_polished_with_local_pcn.builder import (
    gen_maskcnn_polished_with_local_pcn, load_modules
)

from key_utils import keygen

load_modules()


def master(*,
           split_seed,
           model_seed,
           act_fn,
           loss_type,
           input_size,
           out_channel,
           num_layer,
           kernel_size_l1,
           pooling_ksize,
           scale, scale_name,
           smoothness, smoothness_name,
           pooling_type,
           bn_before_act,
           bn_after_fc,

           pcn_cls: int,
           pcn_bypass: bool,
           pcn_no_act: bool,
           pcn_bn_post: bool,
           pcn_final_act: bool,
           pcn_bn: bool,
           pcn_bias: bool,
           ):
    key = keygen(
        split_seed=split_seed,
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        input_size=input_size,
        out_channel=out_channel,
        num_layer=num_layer,
        kernel_size_l1=kernel_size_l1,
        pooling_ksize=pooling_ksize,
        scale_name=scale_name,
        smoothness_name=smoothness_name,
        pooling_type=pooling_type,
        bn_before_act=bn_before_act,
        bn_after_fc=bn_after_fc,

        pcn_bn=pcn_bn,
        pcn_bn_post=pcn_bn_post,
        pcn_bypass=pcn_bypass,
        pcn_cls=pcn_cls,
        pcn_final_act=pcn_final_act,
        pcn_no_act=pcn_no_act,
        pcn_bias=pcn_bias,
    )

    # keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
    datasets = get_data('a', 200, input_size, ('042318', '043018', '051018'), scale=0.5,
                        seed=split_seed)

    datasets = {
        'X_train': datasets[0].astype(np.float32),
        'y_train': datasets[1],
        'X_val': datasets[2].astype(np.float32),
        'y_val': datasets[3],
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }

    def gen_cnn_partial(input_size_cnn, n):
        return gen_maskcnn_polished_with_local_pcn(
            input_size=input_size_cnn,
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

            pcn_bn=pcn_bn,
            pcn_bn_post=pcn_bn_post,
            pcn_bypass=pcn_bypass,
            pcn_cls=pcn_cls,
            pcn_final_act=pcn_final_act,
            pcn_no_act=pcn_no_act,
            pcn_b0_init=1.0,
            pcn_tied=False,
            pcn_bias=pcn_bias,
            bn_locations_legacy=True,
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
