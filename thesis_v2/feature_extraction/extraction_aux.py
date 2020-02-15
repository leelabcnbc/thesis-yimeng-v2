from os import makedirs
from os.path import dirname

import h5py

from torch import tensor

from .preprocessing import preprocess_dataset_imagenet
from .extraction import extract_features


# noinspection PyCallingNonCallable
#   this is for `preprocessor=lambda x: (tensor(x[0]).cuda(),),`
def process_one_case_wrapper_imagenet(*,
                                      net_name_this, net_this,
                                      dataset_np_this,
                                      grp_name,
                                      setting_this, batch_size,
                                      file_to_save_input, file_to_save_feature,
                                      get_one_network_meta_fn,
                                      deterministic=True,
                                      dataset_grp_name=None,
                                      input_size=(224, 224),
                                      preprocess=True,
                                      flush=False,
                                      compression=True,
                                      ):
    assert setting_this.keys() == {'scale', 'rf_size', 'bg_color'}

    if not preprocess:
        assert setting_this['scale'] is None
        assert setting_this['bg_color'] is None

    # this works for generic imagenet networks trained in PyTorch convention.
    augment_config = get_one_network_meta_fn(
        net_name_this, setting_this['rf_size'])

    if dataset_grp_name is None:
        dataset_grp_name = grp_name
    print(grp_name, augment_config['module_names'])
    print(dataset_grp_name)

    # create dir
    makedirs(dirname(file_to_save_feature), exist_ok=True)
    makedirs(dirname(file_to_save_input), exist_ok=True)

    with h5py.File(file_to_save_feature, 'a') as f_feature:
        if grp_name not in f_feature:
            # then preproces dataset
            with h5py.File(file_to_save_input, 'a') as f_input:
                if dataset_grp_name not in f_input:
                    if preprocess:
                        dataset_np = preprocess_dataset_imagenet(
                            images=dataset_np_this,
                            bgcolor=setting_this['bg_color'],
                            input_size=input_size,
                            rescale_ratio=setting_this['scale'],
                        )
                    else:
                        # this can be useful in certain cases, like testing imagenet val images.
                        dataset_np = dataset_np_this
                    f_input.create_dataset(dataset_grp_name,
                                           data=dataset_np,
                                           compression="gzip")
                    f_input.flush()
                    print(f'{grp_name} input computation done')
                else:
                    print(f'{grp_name} input computation done before!')

            # `r` for safety
            with h5py.File(file_to_save_input, 'r') as f_input:
                # use h5 itself. no pre reading.
                assert dataset_grp_name in f_input
                dataset_preprocessed = f_input[dataset_grp_name]

                print(dataset_preprocessed.shape)

                grp = f_feature.create_group(grp_name)

                extract_features(net_this, (dataset_preprocessed,),
                                 preprocessor=lambda x: (tensor(x[0]).cuda(),),
                                 output_group=grp,
                                 batch_size=batch_size,
                                 augment_config=augment_config,
                                 # mostly for replicating old results
                                 deterministic=deterministic,
                                 flush=flush,
                                 compression=compression,
                                 )
        else:
            print(f'{grp_name} feature extraction done before')
