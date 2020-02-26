from os.path import dirname, relpath, realpath

from thesis_v2 import dir_dict
from thesis_v2.submission import utils

from thesis_v2.configs.model.feature_approximation import (
    model_params_b_kl_recurrent_20200218,
    k_bl_recurrent_sep2_hyparameters,
    consts,
    script_keygen
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
basemodel_key_script={basemodel_key_script},
basemodel_idx={basemodel_idx},
sep='2+'+{sep_start},
)
""".strip()


def good_model_param(param):
    # right now only care about those models with cycle = 5; input_size=50 to reduce output size
    return param['rcnn_bl_cls'] == 4 and param['kernel_size_l23'] == 3 and param['num_layer'] == 3


def param_iterator(sep_start_range=(0, 1)):
    for key_script, basemodel_param_dict in model_params_b_kl_recurrent_20200218(
        good_model_param
    ).items():
        for param_dict in k_bl_recurrent_sep2_hyparameters().generate():
            param_dict_actual = {
                **param_dict,
                **{
                    # for new model's script name and folder name
                    'basemodel_idx': basemodel_param_dict['idx'],
                    # for loading base model response
                    'basemodel_key_script': key_script,
                }
            }

            # different starting point.
            # need to be encoded in model prefix
            for sep_start in sep_start_range:
                key_this, key_this_original = script_keygen(
                    return_original_key=True,
                    return_model_prefix=True,
                    **{
                    **param_dict_actual,
                    # add model_prefix
                    **{
                        'model_prefix': consts[f'k_bl_recurrent_k3_sep2+{sep_start}_model_prefix'],
                    },
                })

                yield {
                    'param_dict_actual': param_dict_actual,
                    'key_this': key_this,
                    'key_this_original': key_this_original,
                    'sep_start': sep_start,
                }


def main():
    script_dict = dict()

    for data in param_iterator():
        key_this = data['key_this']
        param_dict_actual = data['param_dict_actual']
        sep_start = data['sep_start']

        script_dict[key_this] = utils.call_script_formatter(
            call_script, set(),
            current_dir=current_dir,
            **param_dict_actual,
            sep_start=str(sep_start),
        )

    utils.submit(
        # the transfer learning one is fine.
        script_dict, 'maskcnn_like', 'standard', True,
        dirname_relative='scripts+yuanyuan_8k_a_3day+feature_approximation_k_bl_recurrent_k3_sep2'  # noqa: E501
    )


if __name__ == '__main__':
    main()
