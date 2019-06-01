import numpy as np
from skimage.transform import rescale

from leelabtoolbox.preprocessing import pipeline


def preprocess_dataset_imagenet(
        *,
        images, bgcolor, input_size, rescale_ratio=None
):
    # this is ready for all torchvision style imagenet network

    # late calling, this is good for some large data set,
    # like 8k
    if not isinstance(images, np.ndarray):
        assert callable(images)
        images = images()

    print(images.shape)

    # rescale
    if rescale_ratio is not None:
        images = np.asarray(
            # changed to suppress some skimage warnings, and be explicit
            # about the last dim's semantics.
            [rescale(im, scale=rescale_ratio, order=1, mode='edge',
                     anti_aliasing=False,
                     multichannel=True if im.ndim == 3 else False) for im in
             images])

    # make sure images are 3D
    if images.ndim == 3:
        images = np.concatenate((images[..., np.newaxis],) * 3, axis=-1)
    assert images.ndim == 4 and images.shape[-1] == 3
    assert np.all(images <= 1) and np.all(images >= 0)

    # use leelab-toolbox pipeline
    steps_naive = ['putInCanvas']
    pars_naive = {'putInCanvas': {'canvas_size': input_size,
                                  'canvas_color': bgcolor,
                                  },
                  }
    (pipeline_naive, realpars_naive,
     order_naive) = pipeline.preprocessing_pipeline(
        steps_naive, pars_naive,
        order=steps_naive)
    images_new = pipeline_naive.transform(
        images.astype(np.float32, copy=False))

    # normalize
    # check
    # http://pytorch.org/docs/master/torchvision/models.html
    images_new -= np.array([0.485, 0.456, 0.406])
    images_new /= np.array([0.229, 0.224, 0.225])
    # transpose
    images_new = np.transpose(images_new, (0, 3, 1, 2))
    # done
    return images_new
