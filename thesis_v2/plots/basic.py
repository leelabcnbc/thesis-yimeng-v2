from typing import List, Tuple
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
        scatter_s=1,
        xlim=(0, 1),
        ylim=(0, 1),
        set_axis_equal=True,
        remove_x_axis_labels=False,
        remove_y_axis_labels=False,
        plot_equal_line=True,
        label=None
):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    if label is None:
        ax.scatter(x, y, s=scatter_s)
    else:
        ax.scatter(x, y, s=scatter_s, label=label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if set_axis_equal:
        ax.axis('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if remove_x_axis_labels:
        ax.axes.xaxis.set_ticklabels([])
    if remove_y_axis_labels:
        ax.axes.yaxis.set_ticklabels([])

    if plot_equal_line:
        ax.plot([0, 1], [0, 1], linestyle='--', color='k')


def labeled_line(
        ax: Axes,
        data: List[Tuple[str, np.ndarray]],
        xticks: List[str],
        *,
        xlabel,
        ylabel,
        xlim_offset=(-0.5, 0.5),
        ylim=None,
        plot_kwargs=None,
        xticklabels_kwargs=None,
):
    if plot_kwargs is None:
        plot_kwargs = dict()
    for label, data_this in data:
        assert data_this.ndim == 1
        ax.plot(np.arange(data_this.size), data_this,
                label=label,
                **plot_kwargs
                )

    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks, **xticklabels_kwargs)
    ax.set_xlim(0 + xlim_offset[0], len(xticks) - 1 + xlim_offset[1])
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
