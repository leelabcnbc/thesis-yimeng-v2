"""handle image preprocessing for feature extraction

this is code to handle local PCN networks (Purdue implementation).
"""

# mostly based on https://github.com/leelabcnbc/thesis-yimeng-v1/blob/7faf932949f0ea7268f0a6cbd7c9c4420e0e2b25/scripts/debug/pcn_local/check_RF_size.ipynb  # noqa: E501

from . import meta
from ...feature_extraction.extraction_aux import (
    process_one_case_wrapper_imagenet
)


def get_one_network_meta(net_name, rf_size):
    # get meta info needed for this network
    correspondence_func = meta.get_name_mapping(net_name)
    slicing_dict = meta.get_slicing_dict(model_name=net_name,
                                         rf_size=rf_size)
    blobs_to_extract = meta.get_layers_to_extract(net_name)

    return {
        'module_names': blobs_to_extract,
        'name_mapping': correspondence_func,
        'slice_dict': slicing_dict,
    }


def process_one_case_wrapper(*,
                             net_name_this, net_this,
                             dataset_np_this,
                             grp_name,
                             setting_this, batch_size,
                             file_to_save_input,
                             file_to_save_feature,
                             dataset_grp_name
                             ):
    return process_one_case_wrapper_imagenet(
        net_name_this=net_name_this,
        net_this=net_this,
        dataset_np_this=dataset_np_this,
        grp_name=grp_name,
        setting_this=setting_this,  # has scale, bg_color, and rf_size.
        batch_size=batch_size,
        file_to_save_input=file_to_save_input,
        file_to_save_feature=file_to_save_feature,
        get_one_network_meta_fn=get_one_network_meta,
        dataset_grp_name=dataset_grp_name,
    )
