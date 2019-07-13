import numpy as np
from torch.backends import cudnn
import skimage

from thesis_v2.models.vgg.feature_extraction import (
    process_one_case_wrapper
)

from thesis_v2.models.vgg.loader import get_pretrained_network

from thesis_v2 import dir_dict, join
from thesis_v2.data.raw import load_data

# this can save memory error.
# See <https://github.com/pytorch/pytorch/issues/1230>
cudnn.benchmark = False
cudnn.enabled = True


def load_image_dataset(image_dataset_key):
    # hacked from https://github.com/leelabcnbc/tang_jcompneuro_revision/blob/master/tang_jcompneuro/io.py  # noqa: E501
    assert image_dataset_key in {'large', 'medium'}
    image_data = load_data('crcns_pvc-8_images', image_dataset_key)[:540]
    assert image_data.shape == (540, 320, 320)

    assert image_data.dtype == np.uint8
    # otherwise, images will not be normalized correctly.
    image_data = skimage.img_as_float(image_data)  # normalize data
    assert np.all(image_data >= 0.0) and np.all(image_data <= 1.0)
    assert image_data.ndim == 3
    return image_data


# just as an example.
networks_to_try = (
    # to compare against siming's stuff.
    'vgg16',
    'vgg16_bn',
    # # to compare against yuanyuan's stuff.
    # 'vgg19',
    # 'vgg19_bn',
    # 'vgg13',
    # 'vgg13_bn',
    'vgg11',
    'vgg11_bn',
)

settings_dict = {
    # # according to the paper, 150px corrsponds to 3.1 deg.
    #     # so roughly, 144 for 3 deg.
    # so 48 pixels in original space should be 1 deg.
    # in the half scale, 24 pixel should be one degree roughly.
    # by checking the input result, it's correct.

    'half': {'scale': 1 / 2, 'rf_size': 24},
    'one_third': {'scale': 1 / 3, 'rf_size': 16},
    'two_third': {'scale': 2 / 3, 'rf_size': 32},
    'five_sixth': {'scale': 5 / 6, 'rf_size': 40},
    'one': {'scale': 1.0, 'rf_size': 48},

    # matching amount of info in the data-driven version.
    'half_full': {'scale': 1 / 2, 'rf_size': 72},
    'quarter_full': {'scale': 1 / 4, 'rf_size': 36},
}

bg_dict_per_dataset = {
    # background color.
    'large': np.array([116.0 / 255, ] * 3),
    'medium': np.array([116.0 / 255, ] * 3),
}

dataset_dict = {
    'large': lambda: load_image_dataset('large'),
    'medium': lambda: load_image_dataset('medium'),
}

# I name them with a, just to separate files for a,b,c,
# for future. since these files can be huge.

# this file can be used across all networks, as they all take 224x224 input,
# and use pytorch-style imagenet preprocessing.
file_to_save_input = join(dir_dict['datasets'],
                          'cnn_feature_extraction_input',
                          'crcns_pvc8.hdf5')

file_to_save_feature = join(dir_dict['features'],
                            'cnn_feature_extraction',
                            'crcns_pvc8',
                            'vgg.hdf5'
                            )


def do_all():
    for net_name in networks_to_try:
        net = get_pretrained_network(net_name)
        net.cuda()
        for dataset, dataset_fn in dataset_dict.items():
            # dataset_np = dataset_fn()
            print(dataset)
            for setting_name, setting in settings_dict.items():
                process_one_case_wrapper(
                    net_name_this=net_name,
                    net_this=net,
                    dataset_np_this=dataset_fn,
                    grp_name=f'{dataset}/{net_name}/{setting_name}',
                    setting_this={
                        **setting,
                        **{'bg_color': bg_dict_per_dataset[dataset]}
                    },
                    batch_size=50,
                    file_to_save_input=file_to_save_input,
                    file_to_save_feature=file_to_save_feature,
                    dataset_grp_name=f'{dataset}/{setting_name}'
                )
        # may save some memory.
        del net


if __name__ == '__main__':
    do_all()
