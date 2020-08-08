from typing import Optional
import numpy as np

from thesis_v2.data.prepared.gaya import get_data, global_dict
from thesis_v2.analysis.postprocessing import master as master_inner


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
           ff_1st_block: bool,
           ff_1st_bn_before_act: bool,
           kernel_size_l23: int,
           train_keep: int,
           model_prefix: str,
           yhat_reduce_pick: int,
           dataset_prefix: str,
           accumulator_mode: str,
           ):
    datasets = get_data(seed=split_seed, scale=0.5)

    assert train_keep is not None
    assert train_keep <= global_dict['legacy_num_img_train']
    train_keep_slice = slice(train_keep)

    datasets = {
        'X_train': datasets[0][train_keep_slice].astype(np.float32),
        'y_train': datasets[1][train_keep_slice],
        'X_val': datasets[2].astype(np.float32),
        'y_val': datasets[3],
        'X_test': datasets[4].astype(np.float32),
        'y_test': datasets[5],
    }

    master_inner(
        split_seed=split_seed,
        model_seed=model_seed,
        act_fn=act_fn,
        loss_type=loss_type,
        input_size=input_size,
        out_channel=out_channel,
        num_layer=num_layer,
        kernel_size_l1=kernel_size_l1,
        pooling_ksize=pooling_ksize,
        scale=scale, scale_name=scale_name,
        smoothness=smoothness, smoothness_name=smoothness_name,
        pooling_type=pooling_type,
        bn_after_fc=bn_after_fc,
        rcnn_bl_cls=rcnn_bl_cls,
        rcnn_bl_psize=rcnn_bl_psize,
        rcnn_bl_ptype=rcnn_bl_ptype,
        rcnn_acc_type=rcnn_acc_type,
        ff_1st_block=ff_1st_block,
        ff_1st_bn_before_act=ff_1st_bn_before_act,
        kernel_size_l23=kernel_size_l23,
        train_keep=train_keep,
        model_prefix=model_prefix,
        yhat_reduce_pick=yhat_reduce_pick,
        dataset_prefix=dataset_prefix,
        accumulator_mode=accumulator_mode,
        datasets=datasets,
    )
