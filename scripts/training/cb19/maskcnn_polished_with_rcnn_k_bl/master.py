from typing import Optional

import numpy as np

from thesis_v2.data.prepared.cadena_plos_cb19 import get_data, global_dict

from thesis_v2.training_extra.maskcnn_like.opt import get_maskcnn_v1_opt_config
from thesis_v2.training_extra.maskcnn_like.training import (train_one,
                                                            partial)

from thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder import (
    gen_maskcnn_polished_with_rcnn_k_bl, load_modules
)

from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    keygen
)

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

           bn_after_fc,

           rcnn_bl_cls: int,
           rcnn_bl_psize: int,
           rcnn_bl_ptype: Optional[str],
           rcnn_acc_type: str,

           dataset_prefix: str,
           # sync with thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder.gen_maskcnn_polished_with_rcnn_k_bl
           ff_1st_block: bool = False,
           ff_1st_bn_before_act: bool = True,
           kernel_size_l23: int = 3,
           train_keep: Optional[int] = None,

           seq_length: Optional[int] = None,
           val_test_every: Optional[int] = None,
           show_every: int = 100,
           px_kept: Optional[int] = None,
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
        kernel_size_l23=kernel_size_l23,
        pooling_ksize=pooling_ksize,
        scale_name=scale_name,
        smoothness_name=smoothness_name,
        pooling_type=pooling_type,
        bn_after_fc=bn_after_fc,
        rcnn_bl_cls=rcnn_bl_cls,
        rcnn_bl_psize=rcnn_bl_psize,
        rcnn_bl_ptype=rcnn_bl_ptype,
        rcnn_acc_type=rcnn_acc_type,

        ff_1st_block=ff_1st_block,
        ff_1st_bn_before_act=ff_1st_bn_before_act,
        train_keep=train_keep,
        dataset_prefix=dataset_prefix,
        seq_length=seq_length,
        px_kept=px_kept,
    )

    print('key', key)

    if px_kept is None:
        px_kept = 80

    if seq_length is None:
        # keeping mean response at 0.5 seems the best. somehow. using batch norm is bad, somehow.
        datasets = get_data(px_kept=px_kept, final_size=input_size, seed=split_seed, scale=0.5)
    else:
        raise ValueError

    if train_keep is not None:
        assert train_keep <= global_dict['num_img']*0.8*0.8
        train_keep_slice = slice(train_keep)
    else:
        train_keep_slice = slice(None)

    datasets = {
        'X_train': datasets[0][train_keep_slice].astype(np.float32),
        'y_train': datasets[1][train_keep_slice],
        'X_val': datasets[2].astype(np.float32),
        'y_val': datasets[3],
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }

    for z in datasets:
        print(z, datasets[z].shape)


    def gen_cnn_partial(input_size_cnn, n):
        return gen_maskcnn_polished_with_rcnn_k_bl(
            input_size=input_size_cnn,
            num_neuron=n,
            out_channel=out_channel,
            kernel_size_l1=kernel_size_l1,  # (try 5,9,13)
            kernel_size_l23=kernel_size_l23,
            act_fn=act_fn,
            pooling_ksize=pooling_ksize,  # (try, 1,3,5,7)
            pooling_type=pooling_type,  # try (avg, max)  # looks that max works well here?
            num_layer=num_layer,
            bn_after_fc=bn_after_fc,
            n_timesteps=rcnn_bl_cls,
            blstack_pool_ksize=rcnn_bl_psize,
            blstack_pool_type=rcnn_bl_ptype,
            acc_mode=rcnn_acc_type,
            factored_constraint=None,
            ff_1st_block=ff_1st_block,
            ff_1st_bn_before_act=ff_1st_bn_before_act,
            num_input_channel=(1 if seq_length is None else seq_length)
        )

    opt_config_partial = partial(
        get_maskcnn_v1_opt_config,
        scale=scale,
        smoothness=smoothness,
        group=0.0,
        loss_type=loss_type,
    )

    if val_test_every is not None:
        added_kw = {
            'val_test_every': val_test_every,
        }
    else:
        added_kw = dict()
    added_kw['show_every'] = show_every

    result = train_one(
        arch_json_partial=gen_cnn_partial,
        opt_config_partial=opt_config_partial,
        datasets=datasets,
        key=key,
        max_epoch=40000,
        model_seed=model_seed,
        return_model=False,
        extra_params={
            # reduce on batch axis
            'eval_fn': {'yhat_reduce_axis': 1}
        },
        **added_kw,
    )

    print(result['stats_best']['stats']['test']['corr_mean'])
