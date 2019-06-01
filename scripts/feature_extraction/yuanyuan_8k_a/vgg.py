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
    assert image_dataset_key in {'a/images'}
    image_data = load_data('yuanyuan_8k_images', image_dataset_key)
    assert image_data.shape == (8000, 400, 400)
    # normalize data
    image_data = skimage.img_as_float(image_data)
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
    # 200*1/4 = 50
    'quarter': {'scale': 1 / 4, 'rf_size': 50},
    # 200*1/2 = 100
    # may help.
    'half': {'scale': 1 / 2, 'rf_size': 100},
}

bg_dict_per_dataset = {
    # background color.
    'a': np.array([121.0 / 255, ] * 3),
}

dataset_dict = {
    'a': lambda: load_image_dataset('a/images'),
}

# I name them with a, just to separate files for a,b,c,
# for future. since these files can be huge.

# this file can be used across all networks, as they all take 224x224 input,
# and use pytorch-style imagenet preprocessing.
file_to_save_input = join(dir_dict['datasets'],
                          'cnn_feature_extraction_input',
                          'yuanyuan_8k_a.hdf5')

file_to_save_feature = join(dir_dict['features'],
                            'cnn_feature_extraction',
                            'yuanyuan_8k_a',
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
