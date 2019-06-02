from os.path import join
from joblib import dump, load


# from <https://github.com/pytorch/examples/blob/c5985a81a8be892ef6c5bd4ace734fb0f18cad73/imagenet/main.py#L356-L371>  # noqa: E501
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_memmapped_array(arr_list, dir_to_dump):
    # based on https://joblib.readthedocs.io/en/latest/parallel.html#manual-management-of-memmaped-input-data  # noqa: E501
    arr_list_new = []
    for idx, arr in enumerate(arr_list):
        f_this = join(dir_to_dump, f'{idx}.mmap')
        dump(arr, f_this)
        arr_list_new.append(load(f_this, mmap_mode='r'))
    return arr_list_new
