from os.path import dirname, relpath, realpath

from thesis_v2 import dir_dict
from thesis_v2.submission import utils

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
shuffle_type={shuffle_type},
split_seed={split_seed},
model_seed={model_seed},
act_fn={act_fn},
loss_type={loss_type},
out_channel={out_channel},
num_layer={num_layer},
kernel_size_l1={kernel_size_l1},
pooling_ksize={pooling_ksize},
scale_name={scale_name}, scale={scale},
smoothness_name={smoothness_name}, smoothness={smoothness},
pooling_type={pooling_type},
bn_before_act={bn_before_act},
bn_after_fc={bn_after_fc},
)
""".strip()

param_iterator_obj = utils.ParamIterator()

param_iterator_obj.add_pair(
    'shuffle_type',
    ('legacy', 'paired')
)

param_iterator_obj.add_pair(
    'split_seed',
    # also try some other splits, with each class represented equally.
    range(3)
)

param_iterator_obj.add_pair(
    'model_seed',
    # range(5),
    range(3),  # otherwise too long.
)

param_iterator_obj.add_pair(
    'act_fn',
    # should try relu later
    ('softplus', 'relu'),
)

param_iterator_obj.add_pair(
    'loss_type',
    ('poisson', 'mse')  # should try mse later
)

param_iterator_obj.add_pair(
    'out_channel',
    (16,)
)

param_iterator_obj.add_pair(
    'num_layer',
    (2,)
)

param_iterator_obj.add_pair(
    'kernel_size_l1',
    (7,)
)

param_iterator_obj.add_pair(
    'pooling_ksize',
    (3,)
)

param_iterator_obj.add_pair(
    'pooling_type',
    ('avg',)
)

param_iterator_obj.add_pair(
    'bn_before_act',
    (True, False)  # should try False later
)

param_iterator_obj.add_pair(
    'bn_after_fc',
    (False, True)  # should try True later
)

param_iterator_obj.add_pair(
    ('scale_name', 'scale'),
    lambda: {
        # key is the name, value is the actual value to be passed in as is.
        '0.01105': '0.00005 * 221',
    }.items(),
    late_call=True,
)

param_iterator_obj.add_pair(
    ('smoothness_name', 'smoothness'),
    lambda: {
        '0.00002': '0.00002',
    }.items(),
    late_call=True,
)


def main():
    # this way, I can import this submit file normally if I need.
    from key_utils import script_keygen

    script_dict = dict()
    for param_dict in param_iterator_obj.generate():
        key_this = script_keygen(**param_dict)

        script_dict[key_this] = utils.call_script_formatter(
            call_script, {'smoothness', 'scale'},
            current_dir=current_dir,
            **param_dict,
        )

    utils.submit(
        script_dict, 'maskcnn_like', 'standard', True,
        dirname_relative='scripts+crcns_pvc8_large+maskcnn_polished'  # noqa: E501
    )


if __name__ == '__main__':
    main()
