# plot ablation study data.
# based on 20201221_plot.ipynb as well as
# 20201205+20201205_2+20201213+20201213_2_plot.ipynb
# under /results_processed/yuanyuan_8k_a_3day_refactored/maskcnn_polished_with_rcnn_k_bl/.

from os.path import join
from os import makedirs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .util import savefig

def plot_all(*, df: pd.DataFrame, plot_dir: str, ff_perf=None):
    makedirs(plot_dir, exist_ok=True)
    for aggregate_level in df.index.get_level_values('aggregate_level').unique():
        if aggregate_level == '':
            title_suffix = 'all readout modes'
        else:
            title_suffix = aggregate_level.split('=')[1]

        plot_one(
            df = df.xs(aggregate_level, level='aggregate_level').sort_index(),
            plot_dir = plot_dir,
            title_suffix=title_suffix,
            ff_perf=ff_perf,
        )

def plot_one(
        *,
        df: pd.DataFrame,
        plot_dir: str,
        title_suffix: str,
        ff_perf
):
    # first plot,
    # perf vs nomimal depth.
    # this only involves 'multipath' and 'leDXgeDX'
    plot_perf_change_plot(
        df = df,
        plot_dir = plot_dir,
        title_suffix = title_suffix,
        ff_perf=ff_perf,
    )

    # second plot.
    # the real thing.
    plot_perf_vs_depth_plot(
        df = df,
        plot_dir = plot_dir,
        title_suffix = title_suffix,
        ff_perf=ff_perf,
    )

def plot_perf_change_plot(
    *,
        df: pd.DataFrame,
        plot_dir: str,
        title_suffix: str,
        ff_perf,
):
    unique_cls = df.index.get_level_values('rcnn_bl_cls').unique()
    assert np.array_equal(unique_cls, np.arange(2, unique_cls[-1] + 1))
    max_cls = unique_cls[-1]

    plt.close('all')
    fig, ax = plt.subplots(1,1,squeeze=True, figsize=(6, 4))
    n_all = []
    for cls_this in unique_cls:

        data_ablation = df.xs('leDXgeDX', level='data_source').loc[cls_this]
        data_baseline = df.xs('multipath', level='data_source').loc[cls_this]
        n = data_ablation['n']
        assert n == data_ablation['n'] == data_baseline['n']
        n_all.append(n)

        assert len(data_ablation['perf_mean']) == 2*(cls_this-1)
        assert len(data_ablation['perf_sem']) == 2 * (cls_this - 1)

        data_all_to_plot_mean = []
        data_all_to_plot_err = []

        data_all_to_plot_mean.extend(data_ablation['perf_mean'][:cls_this-1])
        data_all_to_plot_err.extend(data_ablation['perf_sem'][:cls_this-1])

        assert len(data_baseline['perf_mean']) == 1
        assert len(data_baseline['perf_sem']) == 1

        data_all_to_plot_mean.extend(data_baseline['perf_mean'][:cls_this - 1])
        data_all_to_plot_err.extend(data_baseline['perf_sem'][:cls_this - 1])

        data_all_to_plot_mean.extend(data_ablation['perf_mean'][cls_this - 1:])
        data_all_to_plot_err.extend(data_ablation['perf_sem'][cls_this - 1:])

        assert len(data_all_to_plot_mean) == len(data_all_to_plot_err) == 2 * cls_this - 1

        offset = max_cls - cls_this
        # so that every one has original perf in the middle.
        ax.errorbar(
            x=np.arange(2 * (cls_this) - 1) + offset,
            y=np.array(data_all_to_plot_mean),
            yerr=np.array(data_all_to_plot_err), label=str(cls_this)
        )

    ax.legend(loc='best', title='number of iterations T')
    ax.set_ylabel('''Average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$''')
    ax.set_xticks(np.arange(2 * (max_cls) - 1))
    ax.set_xlabel('''kept path length |p|''')
    ax.set_xticklabels(
        # TODO: fix this for num_layer = 3 models.
        [f'|p|≤T-{d_lower}' for d_lower in range(1, max_cls)[::-1]] +
        ['|p|=1,...,T'] +
        [f'|p|≥{d_higher}' for d_higher in range(2, max_cls + 1)], rotation=45
    )
    ax.text(
        0, 1, s='{} original models across all iterations, {}'.format(sum(n_all), title_suffix), horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
    )
    # ax.axvline(x=cls_this-1, linestyle='--', color='k')
    if ff_perf is not None:
        ax.axhline(y=ff_perf, linestyle='--', color='k')
    fig.subplots_adjust(left=0.125, right=0.99, bottom=0.225, top=0.99)
    savefig(fig, join(plot_dir, 'perf_change_{}.pdf'.format(title_suffix.replace(' ', '+'))))
    plt.show()


def plot_perf_vs_depth_plot(
    *,
        df: pd.DataFrame,
        plot_dir: str,
        title_suffix: str,
        ff_perf,
):
    plt.close('all')
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(6, 4))

    label_mapper = {
        'leDXgeDX': 'longer/shorter paths',
        'onlyDX': 'single length',
        'multipath': 'multi path',
        'original': 'original'
    }

    for data_source in df.index.get_level_values('data_source').unique():
        df_this = df.xs(data_source, level='data_source').sort_index()
        assert np.array_equal(df_this.index.values, np.arange(2, df_this.index.values[-1]+1))
        # depth
        depth_all = np.asarray(sum(df_this['depth_mean'].values.tolist(), [])) + 1
        perf_all = np.asarray(sum(df_this['perf_mean'].values.tolist(), []))

        # filter = depth_all > 1
        filter = slice(None)
        ax.scatter(
            x=depth_all[filter],
            y=perf_all[filter],
            label=label_mapper[data_source],
            s=6,
            marker={
                'leDXgeDX': '^',
                'onlyDX': 'v',
                'multipath': '^',
                'original': '*'
            }[data_source]
        )
    ax.text(
        0, 1, s=title_suffix,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
    )
    ax.legend(loc='best')
    ax.set_ylabel('''Average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$''')
    ax.set_xlabel('''Average path length''')
    if ff_perf is not None:
        ax.axhline(y=ff_perf, linestyle='--', color='k')
    fig.subplots_adjust(right=0.99, top=0.99)
    savefig(fig, join(plot_dir, 'scatter_{}.pdf'.format(title_suffix.replace(' ', '+'))))
    plt.show()
