from os.path import dirname, relpath, realpath
from copy import deepcopy

from thesis_v2 import dir_dict
from thesis_v2.submission import utils
from thesis_v2.configs.model.maskcnn_polished_with_rcnn_k_bl import (
    explored_models_20200914_tang_generator,
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
bn_after_fc={bn_after_fc},
# RCNN-BL specific stuffs.
rcnn_bl_cls={rcnn_bl_cls},
rcnn_bl_psize={rcnn_bl_psize},
rcnn_bl_ptype={rcnn_bl_ptype},
rcnn_acc_type={rcnn_acc_type},
# first layer
ff_1st_block={ff_1st_block},
ff_1st_bn_before_act={ff_1st_bn_before_act},
kernel_size_l23={kernel_size_l23},
train_keep={train_keep},
dataset_prefix={dataset_prefix},
yhat_reduce_pick={yhat_reduce_pick},
additional_key={additional_key},
)
""".strip()


def main():
    script_dict = dict()
    for param_dict in explored_models_20200914_tang_generator():
        param_dict = deepcopy(param_dict)
        del param_dict['model_prefix']
        key_this = script_keygen(**param_dict)
        assert key_this not in script_dict
        script_dict[key_this] = utils.call_script_formatter(
            call_script, {'smoothness', 'scale'},
            current_dir=current_dir,
            **param_dict,
        )

    utils.submit(
        script_dict, 'maskcnn_like', 'standard', True,
        dirname_relative='scripts+tang+maskcnn_polished_with_rcnn_k_bl+20200914'
    )


if __name__ == '__main__':
    main()
