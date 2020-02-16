from os.path import dirname, relpath, realpath

from thesis_v2 import dir_dict
from thesis_v2.submission import utils

from thesis_v2.configs.model.feature_approximation import (
    consts,
    script_keygen,
    local_pcn_original_imagenet_sep2_hyparameters
)

# this is used to find master.py
# you can't drop relpath; this file runs in host, but call_script runs in
# singularity.
# print(dir_dict)
current_dir = realpath(dirname(__file__))
# print(current_dir, dir_dict['root'])
current_dir = relpath(current_dir, dir_dict['root'])
# print(current_dir)

call_script = """
from sys import path
from thesis_v2 import dir_dict, join
path.insert(0, join(dir_dict['root'], {current_dir}))
from master import master

master(
model_seed={model_seed},
act_fn={act_fn},
loss_type={loss_type},
kernel_size={kernel_size},
bn_pre={bn_pre},
basemodel_idx={basemodel_idx},
dataset_prefix={dataset_prefix},
sep=2,
)
""".strip()


def param_iterator():
    # conv0 --- conv10
    for conv_idx in range(11):
        for param_dict in local_pcn_original_imagenet_sep2_hyparameters().generate():
            param_dict_actual = {
                **param_dict,
                **{
                    # for new model's script name and folder name
                    'basemodel_idx': conv_idx,
                    # for loading base model response
                    'dataset_prefix': consts[
                                          'local_pcn_original_imagenet_imagenet_val'
                                      ]['dataset_prefix_prefix'] + '+first500+everything',

                }
            }

            key_this, key_this_original = script_keygen(
                return_original_key=True,
                **{
                    **param_dict_actual,
                    **{
                        'model_prefix': consts['local_pcn_original_imagenet_sep2_model_prefix'],
                        # needed for script_keygen
                        'basemodel_key_script': None,
                    }
                })
            # for debugging.
            print(key_this_original)
            yield {
                'param_dict_actual': param_dict_actual,
                'key_this': key_this,
                'key_this_original': key_this_original,
            }


def main():
    script_dict = dict()

    for data in param_iterator():
        key_this = data['key_this']
        param_dict_actual = data['param_dict_actual']

        script_dict[key_this] = utils.call_script_formatter(
            call_script, set(),
            current_dir=current_dir,
            **param_dict_actual,
        )

    utils.submit(
        # the transfer learning one is fine.
        script_dict, 'maskcnn_like', 'standard', True,
        dirname_relative='scripts+imagenet_val+feature_approximation_lpcn_original_sep2'  # noqa: E501
    )


if __name__ == '__main__':
    main()
