from thesis_v2 import dir_dict, join
from sys import path
from master import master

master(
split_seed='legacy',
model_seed=0,
act_fn='relu',
loss_type='mse',
input_size=50,
out_channel=16,
num_layer=2,
kernel_size_l1=9,
pooling_ksize=3,
scale_name='0.01', scale=0.01,
smoothness_name='0.000005', smoothness=0.000005,
pooling_type='avg',
bn_after_fc=False,
# RCNN-BL specific stuffs.
rcnn_bl_cls=1,
rcnn_bl_psize=1,
rcnn_bl_ptype=None,
rcnn_acc_type='cummean',
# first layer
ff_1st_block=True,
ff_1st_bn_before_act=True,
kernel_size_l23=3,
show_every=300,
val_test_every=150,
seq_length=1,
model_prefix='maskcnn_polished_with_rcnn_k_bl_per_trial'
)