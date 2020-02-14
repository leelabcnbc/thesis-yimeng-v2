"""handle image preprocessing for feature extraction

this is code to handle local PCN networks (Purdue implementation).
"""

# mostly based on https://github.com/leelabcnbc/thesis-yimeng-v1/blob/7faf932949f0ea7268f0a6cbd7c9c4420e0e2b25/scripts/debug/pcn_local/check_RF_size.ipynb  # noqa: E501


from .reference import loader
from ...feature_extraction.extraction_aux import process_one_case_wrapper_imagenet


def get_one_network_meta(net_name, ec_size=22):
    # get meta info needed for this network

    net_name_base = net_name[:net_name.index('_')]
    # TODO: move these functions from `loader` to `meta`, as done in VGG,
    #   to be consistent.
    correspondence_func = loader.get_name_mapping(net_name_base)
    slicing_dict = loader.get_slicing_dict(net_name_base, ec_size)
    blobs_to_extract = loader.get_layers_to_extract(net_name_base)

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
                             dataset_grp_name,
                             preprocess):
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
        preprocess=preprocess,
    )
