# adapted from
# https://github.com/leelabcnbc/thesis-yimeng-v2/blob/3e9ba9b7cd8c7e324aeb27b235f3ea865972c3f5/scripts/feature_extraction/yuanyuan_8k_a/maskcnn_polished_with_rcnn_k_bl/20200218.py  # noqa: E501

from os.path import join, dirname
from os import makedirs

from torch import tensor
import h5py

from torchnetjson.builder import build_net
from thesis_v2.training.training_aux import load_training_results
from thesis_v2.feature_extraction.extraction import extract_features


# then process one model by a model
def process_one_model(*, key_script, key, get_data_fn, global_vars, post_process_fn=None):
    # load data set
    data = get_data_fn()

    for dataset_name, dataset in data.items():
        print(f'process {key_script}/{dataset_name}')
        process_one_model_one_dataset(
            key=key, dataset_to_extract=dataset, dataset_name=dataset_name,
            key_script=key_script,
            global_vars=global_vars,
            post_process_fn=post_process_fn,
        )


def process_one_model_one_dataset(*, key, dataset_to_extract, dataset_name, key_script, global_vars,
                                  post_process_fn=None):
    grp_name = dataset_name
    file_to_save = join(global_vars['feature_file_dir'], key_script + '.hdf5')
    augment_config = global_vars['augment_config']
    makedirs(dirname(file_to_save), exist_ok=True)
    with h5py.File(file_to_save, 'a') as f_feature:
        if grp_name not in f_feature:
            # load model later.
            result = load_training_results(key, return_model=False)
            # load twice, first time to get the model.
            model = load_training_results(key, return_model=True, model=build_net(result['config_extra']['model']))[
                'model']

            model.cuda()
            model.eval()

            grp = f_feature.create_group(grp_name)

            extract_features(model, (dataset_to_extract,),
                             preprocessor=lambda x: (tensor(x[0]).cuda(),),
                             output_group=grp,
                             batch_size=256,
                             augment_config=augment_config,
                             # mostly for replicating old results
                             deterministic=True,
                             flush=True,
                             # set to False to be much faster.
                             # setting to True is slow, can trigger some strange h5py bug, and
                             # also not yielding much compression ratio.
                             compression=False,
                             )
        else:
            print('done before!')

    if post_process_fn is not None:
        post_process_fn(
            key_script=key_script,
            dataset_name=dataset_name,
            file_to_save=file_to_save,
            global_vars=global_vars,
        )


def master_one_case(*, key_script, key, global_vars, get_data_fn, post_process_fn=None):
    # do load_modules() before calling this.
    process_one_model(
        # for file name
        key_script=key_script,
        # for fetching model
        key=key,
        get_data_fn=get_data_fn,
        global_vars=global_vars,
        post_process_fn=post_process_fn,
    )
