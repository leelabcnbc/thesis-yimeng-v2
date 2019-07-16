from os import makedirs
from os.path import join, split

from matplotlib.figure import Figure

from .. import dir_dict


def savefig(fig: Figure, key, *, dpi=300):
    # key is something divided by path separator.
    dirname, fname = split(key)
    makedirs(join(dir_dict['plots'], dirname),
             exist_ok=True)
    fig.savefig(join(dir_dict['plots'], key), dpi=dpi)
