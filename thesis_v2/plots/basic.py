# scatter plot, line plot, that kind of stuff.
from matplotlib.axes import Axes
import numpy as np


def scatter(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str,
        ylabel: str,
        *,
        scatter_s=1
):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    ax.scatter(x, y, s=scatter_s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], linestyle='--')
