from typing import Optional
from functools import partial
import numpy as np

from thesis_v2.data.prepared.yuanyuan_8k import get_data

from thesis_v2.feature_extraction_extra.maskcnn_like import master_one_case

from thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder import (
    load_modules
)

from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    keygen,
    gen_feature_extraction_global_vars
)

from thesis_v2.analysis.resp_diff import postprocess_maskcnn_polished_with_rcnn_k_bl

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
           scale_name,
           smoothness_name,
           pooling_type,

           bn_after_fc,

           rcnn_bl_cls: int,
           rcnn_bl_psize: int,
           rcnn_bl_ptype: Optional[str],
           rcnn_acc_type: str,

           # sync with thesis_v2.models.maskcnn_polished_with_rcnn_k_bl.builder.gen_maskcnn_polished_with_rcnn_k_bl
           ff_1st_block: bool = False,
           ff_1st_bn_before_act: bool = True,
           kernel_size_l23: int = 3,
           train_keep: Optional[int] = None,
           model_prefix: str = 'maskcnn_polished_with_rcnn_k_bl',
           seq_length: Optional[int] = None,
           key_script: str,
           batch_key: str,
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
        model_prefix=model_prefix,
        seq_length=seq_length,
    )

    print('key', key)

    if seq_length is None:
        def datasets_fn():
            datasets = get_data('a', 200, input_size, ('042318', '043018', '051018'), scale=0.5,
                                seed=split_seed)

            datasets = {
                # 'train': datasets[0].astype(np.float32),
                # 'val': datasets[2].astype(np.float32),
                # no space for that much.
                'test': datasets[4].astype(np.float32),
            }

            return datasets
    else:
        raise RuntimeError

    master_one_case(
        key_script=key_script,
        key=key,
        global_vars=gen_feature_extraction_global_vars(key=batch_key),
        get_data_fn=datasets_fn,
        post_process_fn=partial(
            postprocess_maskcnn_polished_with_rcnn_k_bl,
            rcnn_bl_cls=rcnn_bl_cls,
        )
    )
