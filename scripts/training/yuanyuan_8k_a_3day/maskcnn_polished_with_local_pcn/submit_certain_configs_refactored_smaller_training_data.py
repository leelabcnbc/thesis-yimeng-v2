from os.path import dirname, relpath, realpath

from thesis_v2 import dir_dict
from thesis_v2.submission import utils
from thesis_v2.configs.model.maskcnn_polished_with_local_pcn import (
    explored_models_summer_2019_certain_20200501,
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
split_seed={split_seed},
model_seed={model_seed},
act_fn={act_fn},
loss_type={loss_type},
input_size={input_size},
out_channel={out_channel},
num_layer={num_layer},
kernel_size_l1={kernel_size_l1},
pooling_ksize={pooling_ksize},
scale_name={scale_name}, scale={scale},
smoothness_name={smoothness_name}, smoothness={smoothness},
pooling_type={pooling_type},
bn_before_act={bn_before_act},
bn_after_fc={bn_after_fc},
# pcn specific stuffs.
pcn_bn={pcn_bn},
pcn_bn_post={pcn_bn_post},
pcn_bypass={pcn_bypass},
pcn_cls={pcn_cls},
pcn_final_act={pcn_final_act},
pcn_no_act={pcn_no_act},
pcn_bias={pcn_bias},
train_keep={train_keep},
)
""".strip()


def main():
    script_dict = dict()
    for param_dict in explored_models_summer_2019_certain_20200501().generate():
        key_this = script_keygen(**param_dict)

        script_dict[key_this] = utils.call_script_formatter(
            call_script, {'smoothness', 'scale'},
            current_dir=current_dir,
            **param_dict,
        )

    utils.submit(
        script_dict, 'maskcnn_like', 'standard', True,
        dirname_relative='scripts+yuanyuan_8k_a_3day+maskcnn_polished_with_local_pcn_certain_configs_smaller_training_data'  # noqa: E501
    )


if __name__ == '__main__':
    main()
