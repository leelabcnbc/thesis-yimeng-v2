from os import makedirs
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from .main_results import metric_dict, readout_type_mapping
from .util import savefig


def do_one_readout_type(
        *,
        df_original,
        df_ablated,
        metric_name,
        figkey,
        readout_mode,
):
    num_case = df_ablated.index.get_level_values('range_size').unique().size
    assert num_case == 3
    # plot everything
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=num_case, figsize=(2 * num_case, 1.75), sharey=True)
    fig.subplots_adjust(left=0.125, right=0.975, bottom=0.2, top=0.8, wspace=0.2, hspace=0.2)
    axes = axes.ravel()
    for ax, range_size_this in zip(axes, df_ablated.index.get_level_values('range_size').unique()):
        # just plot this.
        plot_one_ax(
            ax=ax,
            perf_original=df_original[metric_name],
            perf_ablated=df_ablated[metric_name].xs(range_size_this, level='range_size'),
            title_prefix=(
                f'{range_size_this} path, ' if range_size_this == 1 else
                f'{range_size_this} paths, '
            ),
            range_size=range_size_this,
        )
    fig.text(0.0, 1.0, readout_mode, ha='left', va='top')
    fig.text(0.5, 0.0, 'lengths of removed paths', ha='center', va='bottom')
    fig.text(0.0, 0.5, metric_dict[metric_name], va='center', rotation='vertical', ha='left')

    savefig(fig, figkey + f'_{metric_name}.pdf')

    plt.show()


def plot_one_ax(
        *,
        ax,
        perf_original: pd.Series,
        perf_ablated: pd.Series,
        title_prefix,
        range_size,
):
    data_collect = []
    perf_original = perf_original.dropna()
    n = perf_original.size
    for idx, range_start in enumerate(perf_ablated.index.get_level_values('range_start').unique()):
        assert range_start == idx + 1
        perf_ablated_this = perf_ablated.xs(range_start, level='range_start')
        assert perf_ablated_this.index.names == perf_original.index.names
        assert perf_original.index.isin(perf_ablated_this.index).all()
        perf_ablated_this = perf_ablated_this[perf_original.index]
        assert np.all(np.isfinite(perf_ablated_this.values))
        assert np.all(np.isfinite(perf_original.values))
        data_collect.append(perf_ablated_this.mean())
    ax.plot(range(1, len(data_collect) + 1), data_collect, marker='x')
    ax.set_xticks(range(1, len(data_collect) + 1))
    ax.set_xticklabels(
        [
            str(x) if range_size == 1 else '{}-{}'.format(x, x + range_size - 1) for x in
            range(1, len(data_collect) + 1)
        ]
    )
    ax.set_title(title_prefix + 'n={}'.format(n))
    ax.axhline(perf_original.mean(), color='k', linestyle='--')


def do_all_readout_type(
        *,
        df_original,
        df_ablated,
        metric_name='cc2_normed_avg',
        plot_dir,
):
    makedirs(plot_dir, exist_ok=True)
    for readout_type in df_ablated.index.get_level_values('readout_type').unique():
        print(readout_type)
        do_one_readout_type(
            df_original=df_original.xs(readout_type, level='readout_type'),
            df_ablated=df_ablated.xs(readout_type, level='readout_type'),
            metric_name=metric_name,
            figkey = join(plot_dir, f'ablation_7_{readout_type}'),
            readout_mode=readout_type_mapping[readout_type]
        )
