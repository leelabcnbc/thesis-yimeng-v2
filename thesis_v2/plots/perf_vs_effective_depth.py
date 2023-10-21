from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..analysis.ablation_study import (
    get_depth,
    get_perf,
    get_depth_entropy,
    get_depth_distribution,
    get_weighted_avg,
)

from .main_results_tables import preprocess
from .main_results import (
    readout_type_order, metric_dict, readout_type_order_mapped, readout_type_mapping
)
from .util import savefig

from os.path import join
from os import makedirs


def get_perf_and_depth(*, df_perf, df_source_analysis, perf_col):
    series_perf = get_perf(df_perf, perf_col)
    series_depth = get_depth(df_source_analysis)
    series_depth_entropy = get_depth_entropy(df_source_analysis)
    series_perf = series_perf.rename('perf')
    series_depth = series_depth.rename('depth')
    series_depth_entropy = series_depth_entropy.rename('entropy')
    assert series_perf.index.equals(series_depth.index)
    assert series_perf.index.equals(series_depth_entropy.index)
    assert np.all(np.isfinite(series_perf.values))
    assert np.all(np.isfinite(series_depth.values))
    assert np.all(np.isfinite(series_depth_entropy.values))

    # add depth distribution.
    series_depth_distribution = get_depth_distribution(df_source_analysis)
    series_depth_distribution = series_depth_distribution.rename('depth_distribution')
    assert series_perf.index.equals(series_depth_distribution.index)

    df_combined = pd.concat(
        [series_perf, series_depth, series_depth_entropy, series_depth_distribution], axis=1
    )
    # everything.

    # across number of layers. this is not available in main results.

    # across channel and number of layers

    ret = dict()

    for agg_type, axes_to_reduce in {
        'all': ['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed',
                'num_layer', 'out_channel'],
        'num_layer': ['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed', 'out_channel'],
        'model_size': ['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed'],
    }.items():
        _dummy, df_ff, df_r, n = preprocess(
            df_in=df_combined,
            axes_to_reduce=axes_to_reduce,
            return_n=True,
            max_cls=None,
        )
        if isinstance(df_ff, pd.DataFrame):
            index_reference_ff = df_r.index.get_loc_level(
                key=(2, 'cm-avg'),
                level=('rcnn_bl_cls', 'readout_type')
            )[1]
            #             print(index_reference_ff)
            assert index_reference_ff.isin(df_ff.index).all()
            df_ff = df_ff.loc[index_reference_ff].sort_index().copy()
        #         print(df_ff)
        # get common index
        ret[agg_type] = {
            'df_ff': df_ff,
            'df_r': df_r,
            'n': n,
        }

    return ret


def get_cls_pick_dict(
        *,
        additional_data,
        y_col,
        data_name,
        setup,
        df_r=None,
):
    if y_col == 'depth_distribution_7':
        return defaultdict(lambda: 7)
    elif y_col == 'depth_distribution_example':
        return additional_data.get(
            data_name, dict()
        ).get(setup, None)
    elif y_col == 'depth_distribution_max':
        assert df_r is not None
        ret = {}
        for readout_type in readout_type_order:
            perf_this = df_r.xs(readout_type, level='readout_type')['perf_mean']
            # get the one with max perf
            ret[readout_type] = perf_this.idxmax()
        return ret
    else:
        raise ValueError


def plot_perf_vs_effective_depth(
        *,
        df_perf,
        df_source_analysis,
        perf_col='cc2_normed_avg',
        df_perf_deep_ff_2layer,
        save_dir,
        additional_data,
        hack_for_nips=False,
):
    makedirs(save_dir, exist_ok=True)
    if df_perf_deep_ff_2layer is not None:
        df_perf_deep_ff_2layer.index = df_perf_deep_ff_2layer.index.droplevel('multi_path_hack')
        df_perf_deep_ff_2layer = df_perf_deep_ff_2layer[perf_col]

    assert df_perf.index.equals(df_source_analysis.index)
    for train_keep in df_perf.index.get_level_values('train_keep').unique():
        print(train_keep)

        training_data_perc = {
            5120: '100%',
            2560: '50%',
            1280: '25%',
            1400: '100%',
            700: '50%',
            350: '25%',
            4640: '100%',
            2320: '50%',
            1160: '25%',
        }[train_keep]
        
        if training_data_perc != '100%':
            continue

        additional_data_this = additional_data.get(train_keep, dict())

        data = get_perf_and_depth(
            df_perf=df_perf.xs(train_keep, level='train_keep'),
            df_source_analysis=df_source_analysis.xs(train_keep, level='train_keep'),
            perf_col=perf_col
        )

        if df_perf_deep_ff_2layer is not None:
            df_perf_deep_ff_2layer_this = df_perf_deep_ff_2layer.xs(train_keep, level='train_keep')


        for data_name, data_values in data.items():
            for y_col in [
                'perf', 'entropy',
                'depth_distribution_7',
                'depth_distribution_example',
                'depth_distribution_max',
            ]:
                xlabel = 'average path length'
                if y_col == 'perf':
                    ylabel = metric_dict[perf_col]
                elif y_col == 'entropy':
                    ylabel = 'path diversity'
                elif y_col.startswith('depth_distribution'):
                    ylabel = 'normalized strength'
                    xlabel = 'path length'
                else:
                    raise ValueError



                plt.close('all')
                if data_name == 'all':
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), squeeze=True)
                    fig.subplots_adjust(left=0.2, right=0.975, hspace=0.2)
                    # just plot this.
                    if not y_col.startswith('depth_distribution'):
                        plot_one_ax(
                            ax=ax,
                            **data_values,
                            xlabel=xlabel,
                            y_col=y_col,
                            ylabel=ylabel,
                            cls_to_pick_dict=get_cls_pick_dict(
                                additional_data=additional_data_this,
                                y_col='depth_distribution_example',
                                data_name=data_name,
                                setup=None,
                            )
                        )
                    else:
                        plot_one_ax_bar(
                            ax=ax,
                            df_r=data_values['df_r'],
                            n=data_values['n'],
                            y_col='depth_distribution',
                            cls_to_pick_dict=get_cls_pick_dict(
                                additional_data=additional_data_this,
                                y_col=y_col,
                                data_name=data_name,
                                setup=None,
                                df_r=data_values['df_r'],
                            )
                        )
                    ax.set_title('All, n={}'.format(data_values['n']))
                else:
                    #                 return data_values
                    if data_name == 'num_layer':
                        # exactly the same layout as in main_tables.py
                        # number of cases should be the same as number of unique pairs in ff's index
                        cases = data_values['df_ff'].index.get_level_values('num_layer').unique().tolist()
                        # it has to be None
                        #
                        level = 'num_layer'
                        assert len(cases) == 2
                        if not (hack_for_nips and y_col.startswith('depth_distribution')):
                            fig, axes = plt.subplots(
                                nrows=1, ncols=2, figsize=(8, 3.5), squeeze=True,
                                sharex=False, sharey=False
                            )
                            fig.subplots_adjust(left=0.125, right=0.975, bottom=0.125, top=0.9, wspace=0.2,
                                                hspace=0.2)
                        else:
                            fig, axes = plt.subplots(
                                nrows=1, ncols=2, figsize=(16, 2), squeeze=True,
                                sharex=False, sharey=False
                            )
                            fig.subplots_adjust(left=0.05, right=0.975, bottom=0.125, top=0.9, wspace=0.2,
                                                hspace=0.2)
                    elif data_name == 'model_size':
                        cases = sorted(
                            set(
                                zip(
                                    data_values['df_ff'].index.get_level_values('out_channel').values.tolist(),
                                    data_values['df_ff'].index.get_level_values('num_layer').tolist()
                                )
                            )
                        )
                        # exactly the same layout as in main_results.py
                        level = ('out_channel', 'num_layer')
                        nrows = (len(cases) - 1) // 2 + 1
                        ncols = 2
                        fig_h = (8 / 3 * nrows) if (nrows != 3) else 8
                        fig, axes = plt.subplots(
                            nrows=nrows, ncols=ncols, figsize=(8, fig_h), squeeze=False,
                            sharex=False, sharey=False
                        )
                        fig.subplots_adjust(
                            left=0.1, right=0.975, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2
                        )
                        axes = axes.ravel()
                    else:
                        raise ValueError

                    fig.text(0.5, 0.0, xlabel, ha='center', va='bottom')
                    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical', ha='left')

                    for idx, setup_this in enumerate(cases):
                        num_variant = data_values['n']

                        if isinstance(setup_this, tuple):
                            assert len(setup_this) == 2
                            num_c = setup_this[0]
                            num_l_ff = setup_this[1]
                            num_l_r = (setup_this[1] - 1) // 2
                            layer_text = 'layer' if num_l_r == 1 else 'layers'
                            title = f'{num_c} ch, {num_l_r} R {layer_text}, n={num_variant}'
                            # title = f'{num_c} ch, 1 C + {num_l_r} RC, n={num_variant}'
                        elif isinstance(setup_this, int):
                            num_l_ff = setup_this
                            num_l_r = (setup_this - 1) // 2
                            if not hack_for_nips:
                                layer_text = 'layer' if num_l_r == 1 else 'layers'
                                title = f'{num_l_r} R {layer_text}, n={num_variant}'
                            # title = f'{num_l_r + 1}L, n={num_variant}'
                            else:
                                layer_text = 'layer'
                                title = f'{num_l_r} R {layer_text} models ({training_data_perc} data), n={num_variant}'
                        else:
                            raise RuntimeError

                        if df_perf_deep_ff_2layer is not None:
                            if data_name == 'num_layer' and setup_this == 3:
                                df_deep_ff = df_perf_deep_ff_2layer_this
                            elif data_name == 'model_size' and setup_this[1] == 3:
                                df_deep_ff = df_perf_deep_ff_2layer_this.xs(setup_this[0], level='out_channel')
                            else:
                                df_deep_ff = None
                        else:
                            df_deep_ff = None

                        ax = axes[idx]

                        if not y_col.startswith('depth_distribution'):
                            plot_one_ax(
                                ax=ax,
                                df_ff=data_values['df_ff'].xs(setup_this,
                                                              level=None if isinstance(level, str) else level),
                                df_r=data_values['df_r'].xs(setup_this, level=level),
                                n=data_values['n'],
                                df_deep_ff=df_deep_ff,
                                y_col=y_col,
                                cls_to_pick_dict=get_cls_pick_dict(
                                    additional_data=additional_data_this,
                                    y_col='depth_distribution_example',
                                    data_name=data_name,
                                    setup=setup_this,
                                )
                            )
                        else:
                            plot_one_ax_bar(
                                ax=ax,
                                df_r=data_values['df_r'].xs(setup_this, level=level),
                                n=data_values['n'],
                                y_col='depth_distribution',
                                cls_to_pick_dict=get_cls_pick_dict(
                                    additional_data=additional_data_this,
                                    y_col=y_col,
                                    data_name=data_name,
                                    setup=setup_this,
                                    df_r=data_values['df_r'].xs(setup_this, level=level),
                                ),
                                num_l_r=num_l_r,
                                hack_for_nips = (
                                    hack_for_nips and level == 'num_layer' and y_col.startswith('depth_distribution')
                                )
                            )

                        if not (
                                    hack_for_nips and level == 'num_layer' and y_col.startswith('depth_distribution')
                                ):

                            ax.set_title(title)
                        else:
                            ax.set_title(title, pad=-10)
                savefig(fig, join(save_dir, f'perf_vs_depth+{train_keep}+{data_name}+{y_col}.pdf'))
                plt.show()


def plot_one_ax(*,
                ax,
                df_ff,
                df_r,
                n,
                df_deep_ff=None,
                y_col,
                xlabel=None,
                ylabel=None,
                cls_to_pick_dict=None,
                ):
    if cls_to_pick_dict is None:
        cls_to_pick_dict = dict()

    if isinstance(df_ff, pd.DataFrame):
        assert df_ff.shape[0] == 1
        df_ff = df_ff.iloc[0]
    assert isinstance(df_ff, pd.Series)
    # follow same read out type order in main results.

    # copied from main_results.py
    margin = (df_r[y_col + '_mean'].max() - df_r[y_col + '_mean'].min()) * 0.05
    dots_to_plot = []
    for readout_type in readout_type_order:
        df_r_this = df_r.xs(readout_type, level='readout_type')
        assert df_r_this.shape[0] == df_r_this.index.get_level_values('rcnn_bl_cls').unique().size
        ax.errorbar(
            df_r_this['depth_mean'] + 1, df_r_this[y_col + '_mean'], label=readout_type,
            marker='x',
            # yerr=df_r_this[y_col + '_sem']
        )

        # use cls_to_pick_dict to highlight certain points.
        dot_to_plot = cls_to_pick_dict.get(readout_type, None)
        if dot_to_plot is not None:
            dots_to_plot.append(
                (
                    df_r_this['depth_mean'].loc[dot_to_plot] + 1,
                    df_r_this[y_col + '_mean'].loc[dot_to_plot],
                )
            )
    # copied from main_results
    ax.legend(
            loc='upper left', ncol=4, bbox_to_anchor=(0.01, 0.99),
            borderaxespad=0., fontsize='x-small', handletextpad=0,
            labels=readout_type_order_mapped,
        )

    # by putting this outside, we do not change the color assignment of main plot
    for dot in dots_to_plot:
        ax.plot(*dot,
                'o', ms=8,
                mec='k',  # edge color
                mfc='none',  # face color
                mew=1,
                )

    if df_deep_ff is not None:
        df_deep_ff = df_deep_ff.unstack('rcnn_bl_cls')
        assert df_deep_ff.shape[0] == n
        df_deep_ff = df_deep_ff.mean(axis=0)
        #         print(df_deep_ff)
        ax.scatter(df_deep_ff.index.values, df_deep_ff.values, label='deep_ff')

    if y_col == 'perf':
        # do not do this for entropy
        ax.axhline(y=df_ff.loc['perf_mean'], linestyle='-', color='k')
        # denote the depth of ff models
        ax.axvline(x=df_ff.loc['depth_mean']+1, linestyle='--', color='k')
        ax.plot(
            df_ff.loc['depth_mean']+1, df_ff.loc['perf_mean'],
            '*', ms=16, mec='k',  # edge color
            mfc='none',  # face color
            mew=2,
        )

    ax.set_ylim(
        df_r[y_col + '_mean'].min() - margin,
        df_r[y_col + '_mean'].max() + 4 * margin,
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_one_ax_bar(
        *,
        ax,
        df_r,
        n,
        y_col,
        xlabel=None,
        ylabel=None,
        cls_to_pick_dict,
        num_l_r=None,
        hack_for_nips=False,
):
    if cls_to_pick_dict is None:
        return

    # construct two new data frames, similar to the ones
    # when I plot the main results.
    # one for mean strength,
    # one for sem strength.

    # these data frames should be constructed
    # by having readout type as rows,
    # and path depth as columns.
    # df_mean = []
    # df_sem = []

    min_val = 0.0
    max_val = 0.0

    # https://stackoverflow.com/a/42091037
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    axis_to_draw_list = []
    for idx, readout_type in enumerate(readout_type_order):
        # drop=False so that my next .xs will work. it's a pandas issue.
        df_r_this = df_r.xs(readout_type, level='readout_type', drop_level=False)
        assert df_r_this.shape[0] == df_r_this.index.get_level_values('rcnn_bl_cls').unique().size
        df_r_this = df_r_this.xs(cls_to_pick_dict[readout_type], level='rcnn_bl_cls')
        mean_this = df_r_this[y_col + '_mean']
        assert mean_this.shape == (1,)
        mean_this = mean_this.iat[0]
        sem_this = df_r_this[y_col + '_sem']
        assert sem_this.shape == (1,)
        sem_this = sem_this.iat[0]
        # length 1 through 8
        assert mean_this.shape == sem_this.shape == (8,)
        avg_length = get_weighted_avg(mean_this)
        max_val = max(max_val, mean_this.max())

        if num_l_r is None:
            # no treatment
            # this is probably for the case `all`
            trim_start = 0
            trim_end = 8
        else:
            # trim this particular one.
            trim_start = {
                1: 0,
                2: 1
            }[num_l_r]
            trim_end = trim_start + cls_to_pick_dict[readout_type]
        if trim_start > 0:
            assert (mean_this[:trim_start] == 0).all()
        if trim_end < 8:
            assert (mean_this[trim_end:] == 0).all()
        mean_this = mean_this[trim_start:trim_end]
        sem_this = sem_this[trim_start:trim_end]

        # add a custom axis.
        if not hack_for_nips:
            axis_to_draw: Axes = ax.inset_axes(
                [
                    *{
                        0: (0.0, 0.5),
                        1: (0.5, 0.5),
                        2: (0.0, 0.0),
                        3: (0.5, 0.0),
                    }[idx],
                    0.5, 0.35
                ]
            )
        else:
            axis_to_draw: Axes = ax.inset_axes(
                [
                    *{
                        0: (0.0, 0.0),
                        1: (0.25, 0.0),
                        2: (0.5, 0.0),
                        3: (0.75, 0.0),
                    }[idx],
                    0.25, 0.8
                ]
            )
        xticks_this = range(1+trim_start, 1+trim_start+mean_this.size)
        axis_to_draw.bar(
            xticks_this,
            mean_this,
            yerr=sem_this,
            color=colors[idx],
        )
        axis_to_draw.set_title(
            # avg_length + 1 is what we really want. the shortest length as a path length of 1, not 0, for humans.
            readout_type_mapping[readout_type] + ', {}, {:.2f}'.format(cls_to_pick_dict[readout_type], avg_length+1)
        )
        axis_to_draw_list.append(axis_to_draw)

        if not hack_for_nips:
            if idx in {1,3}:
                # remove y label
                axis_to_draw.get_yaxis().set_visible(False)
        else:
            if idx in {1,2,3}:
                # remove y label
                axis_to_draw.get_yaxis().set_visible(False)
        axis_to_draw.set_xticks(xticks_this)
        axis_to_draw.set_xticklabels([str(x) for x in xticks_this])

    for ax_this in axis_to_draw_list:
        ax_this.set_ylim(min_val, max_val*1.25)
    ax.axis('off')

    # df_mean = pd.concat(df_mean, axis=0).T
    # df_sem = pd.concat(df_sem, axis=0).T

    # df_mean.plot(
    #     ax=ax, kind='bar', yerr=df_sem, rot=0,
    #     ylim=(min_val, max_val * 1.25),
    # )

    # ax.legend(
    #     loc='upper left', ncol=df_mean.shape[1], bbox_to_anchor=(0.01, 0.99),
    #     borderaxespad=0., fontsize='x-small', handletextpad=0,
    #     #               title='readout type',
    #     labels=readout_type_order_mapped,
    # )
