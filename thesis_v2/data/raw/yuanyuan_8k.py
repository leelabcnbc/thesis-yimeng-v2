from scipy.io import loadmat
import numpy as np
from skimage.io import imread
from . import dir_dict, join, register_data

_data_root = join(dir_dict['private_data'], 'yuanyuan_8k')
_img_root = join(dir_dict['private_data'], 'yuanyuan_8k_raw_images', 'rp_252')


def save_img(group):
    # follow <https://github.com/leelabcnbc/cnn-model-leelab-8000/blob/7d8e86141c3219bc154b7c57960e85b780f70257/leelab_8000/get_images.m>  # noqa: E501
    assert group in {'a', 'b', 'c'}

    recordfile = {
        'a': 'Record_paras_Mar072018_RP.mat',
        'b': 'Record_paras_May072018_RP8001To16000.mat',
        'c': 'Record_paras_May072018_RP16001To24000.mat',
    }[group]

    record = loadmat(join(_data_root, recordfile),
                     variable_names=('Record_paras',))['Record_paras']

    # get image names.
    img_argsort = np.argsort(record[1, 0][0, 1].ravel())
    assert img_argsort.shape == (8000,)

    img_names = sum(
        [[np.string_(str(y[0])) for y in x.ravel()] for x in
         record[1, 0][1:, 1]], [])
    assert len(img_names) == 8000
    img_names = np.asarray(img_names)[img_argsort]
    assert img_names.shape == (8000,)

    # then load actual images.
    images = []
    for name in img_names:
        images.append(imread(join(_img_root, str(name.decode()))))

    images = np.asarray(images)
    assert images.shape == (8000, 400, 400) and images.dtype == np.uint8

    return {
        'images': images,
        # for class and superclasses (merging different classes),
        # all info can be obtained from `names`
        'names': img_names,
    }


register_data('yuanyuan_8k_images', {
    'a': lambda: save_img('a'),
    'b': lambda: save_img('b'),
    'c': lambda: save_img('c'),
})
_neural_data_dates = {'042318', '043018',
                      '050718', '050918',
                      '051018', '051118'}


def save_neural(date):
    assert type(date) is str
    assert date in _neural_data_dates
    file = join(_data_root, 'resp', f'resp_{date}.mat')
    data = loadmat(file, variable_names=['resp'])['resp']
    assert data.dtype == np.float64
    assert data.ndim == 2 and data.shape[0] == 8000

    return {
        # this is what yuanyuan processed.
        # later on we may have other ones.
        'resp': data,
    }


register_data('yuanyuan_8k_neural', {
    date: lambda date=date: save_neural(date) for date in _neural_data_dates
})
