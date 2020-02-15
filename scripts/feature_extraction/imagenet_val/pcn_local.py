import numpy as np

from torch.backends import cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from thesis_v2.models.pcn_local.feature_extraction import (
    process_one_case_wrapper
)

from thesis_v2.models.pcn_local.reference.loader import get_pretrained_network

from thesis_v2 import dir_dict, join

# this can save memory error.
# See <https://github.com/pytorch/pytorch/issues/1230>
cudnn.benchmark = False
cudnn.enabled = True


def load_image_dataset(image_dataset_key):
    assert image_dataset_key == 'first1000'
    torch.manual_seed(0)
    valdir = join('/my_data_2/standard_datasets/ILSVRC2015/Data/CLS-LOC', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        # this batch size should be safe regardless of network type (3CLS or 5CLS)

        # shuffle to make sure we do not get images from the same class.
        batch_size=100, shuffle=True)

    # get data in numpy
    data = []

    for i, (x, _) in enumerate(val_loader):
        data.append(x.numpy())
        if i >= 9:
            break

    data = np.concatenate(data)
    assert data.shape == (1000, 3, 224, 224)
    return data


networks_to_try = (
    'PredNetBpE_3CLS',
)

settings_dict = {
    # using half saves space, plus it avoids boundary effects.
    'center112': {'scale': None, 'rf_size': 112},
}

bg_dict_per_dataset = {
    # background color.
    'first1000': None,
}

dataset_dict = {
    'first1000': lambda: load_image_dataset('first1000'),
}

# I name them with a, just to separate files for a,b,c,
# for future. since these files can be huge.

# this file can be used across all networks, as they all take 224x224 input,
# and use pytorch-style imagenet preprocessing.
file_to_save_input = join(dir_dict['datasets'],
                          'cnn_feature_extraction_input',
                          'imagenet_val.hdf5')

file_to_save_feature = join(dir_dict['features'],
                            'cnn_feature_extraction',
                            'imagenet_val',
                            'pcn_local.hdf5'
                            )


def do_all():
    for net_name in networks_to_try:
        net = get_pretrained_network(
            net_name,
            root_dir=join(
                dir_dict['root'], '..', 'thesis-yimeng-v1', '3rdparty',
                'PCN-with-Local-Recurrent-Processing', 'checkpoint'
            )
        )
        net.cuda()
        for dataset, dataset_fn in dataset_dict.items():
            dataset_np = dataset_fn()
            print(dataset)
            for setting_name, setting in settings_dict.items():
                process_one_case_wrapper(
                    net_name_this=net_name,
                    net_this=net,
                    dataset_np_this=dataset_np,
                    grp_name=f'{dataset}/{net_name}/{setting_name}',
                    setting_this={
                        **setting,
                        **{'bg_color': bg_dict_per_dataset[dataset]}
                    },
                    batch_size=50,
                    file_to_save_input=file_to_save_input,
                    file_to_save_feature=file_to_save_feature,
                    dataset_grp_name=f'{dataset}/{setting_name}',
                    preprocess=False,
                )
        # may save some memory.
        del net


if __name__ == '__main__':
    do_all()
