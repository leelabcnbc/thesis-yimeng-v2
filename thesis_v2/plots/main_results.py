from os.path import join, dirname
from os import makedirs
from typing import List
from itertools import zip_longest, chain, product

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .util import savefig
from .basic import scatter
from .. import dir_dict
from .main_results_tables import get_perf_over_cls_data, get_ff_vs_best_r_data, preprocess, merge_thin_and_wide

metric_dict = {
    'cc2_normed_avg': '''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$''',
    'cc2_raw_avg': '''average ${\\mathrm{CC}}^2$''',
    'cc_raw_avg': '''average $\\mathrm{CC}$''',
}

readout_type_order = ['inst-last', 'cm-last', 'inst-avg', 'cm-avg']
# human readable readout type names in final figures.
readout_type_mapping = {
    'inst-last': 'no-avg',
    'cm-last': 'early-avg',
    'inst-avg': 'late-avg',
    'cm-avg': '2-avg',
}
readout_type_order_mapped = [
    readout_type_mapping[x] for x in readout_type_order
]


def get_r_vs_ff_scatter_inner(
        ax: Axes, perf_ff, perf_r_main, xlabel, ylabel, limit, prefix=None,
        remove_x_axis_labels=False, remove_y_axis_labels=False,
        show_diff_hist=False,
        title=None,
        legend=True,
        show_text=True,
):
    if prefix is None:
        prefix = ''
    merged_main = merge_thin_and_wide(
        df_fewer_columns=perf_r_main, df_more_columns=perf_ff, fewer_suffix=''
    ).sort_index()
    if not show_diff_hist:

        # sep by train_keep and num_layer
        train_keep_all = merged_main.index.get_level_values('train_keep').unique()
        # num_layer_all = merged_main.index.get_level_values('num_layer').unique()
        train_keep_max = train_keep_all.max()
        assert set(train_keep_all) <= {train_keep_max, train_keep_max // 2, train_keep_max // 4}

        for (
                train_keep,
                # num_layer,
        ) in product(
            train_keep_all,
            # num_layer_all
        ):
            # merged_main_this = merged_main.xs((train_keep, num_layer), level=('train_keep', 'num_layer'))
            merged_main_this = merged_main.xs(train_keep, level='train_keep')
            scatter(
                ax=ax,
                x=merged_main_this['perf_ff'].values, y=merged_main_this['perf_r'].values,
                xlabel=xlabel, ylabel=ylabel,
                xlim=limit,
                ylim=limit,
                remove_x_axis_labels=remove_x_axis_labels,
                remove_y_axis_labels=remove_y_axis_labels,
                set_axis_equal=False,
                scatter_s=0.5,
                label=str(100 * train_keep // train_keep_max) + '%',
                plot_equal_line=False
            )
        ax.plot([0, 1], [0, 1], linestyle='--')
        if legend:
            ax.legend(loc='lower right', ncol=1,
                      # borderaxespad=0.,
                      fontsize='small',
                      # handletextpad=0
                      )
    else:
        ax.hist(merged_main['perf_r'].values - merged_main['perf_ff'].values,
                bins=20)
        ax.axvline(x=0, linestyle='--', color='r')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    if show_text:
        ax.text(
            0, 1, s='{}n={}'.format(prefix, merged_main.shape[0]), horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
        )

    if title is not None:
        ax.set_title(title)


def get_r_vs_ff_scatter(df_in, *, max_cls=None, axes_to_reduce, dir_plot, metric,
                        limit=None, deeper_ff=None, show_diff_hist=False
                        ):
    xlabel = metric_dict[metric] + (
        ', FF' if deeper_ff is None else ', deeper FF {}'.format(
            ','.join([str(x) for x in deeper_ff])
        )
    )
    ylabel = metric_dict[metric] + ', recurrent'

    _, df_ff, df_r = preprocess(df_in, max_cls=max_cls, axes_to_reduce=axes_to_reduce)

    perf_ff = df_ff['perf_mean'].to_frame(name='perf_ff')

    if deeper_ff is not None:
        perf_ff = perf_ff[perf_ff.index.get_level_values('num_layer').isin(
            deeper_ff
        )]['perf_ff'].unstack('num_layer').max(axis=1)
        perf_ff = perf_ff.to_frame(name='perf_ff')
        perf_ff['num_layer'] = 3
        perf_ff = perf_ff.set_index('num_layer', append=True)
        suptitle_suffix = '_deeper' + ','.join([str(x) for x in deeper_ff])
    else:
        suptitle_suffix = ''

    if show_diff_hist:
        suptitle_suffix += 'hist'

    perf_r = df_r['perf_mean']
    # main plot, best R vs FF
    total_levels = ('readout_type', 'rcnn_bl_cls')
    perf_r_main = perf_r.unstack(total_levels).max(axis=1).to_frame(name='perf_r')

    if limit is None:
        limit_max, limit_min = max(
            df_r['perf_mean'].quantile(0.975), perf_ff['perf_ff'].quantile(0.975)
        ), min(
            df_r['perf_mean'].quantile(0.025), perf_ff['perf_ff'].quantile(0.025)
        )
        assert limit_max > limit_min
        limit_diff = limit_max - limit_min
        limit_max = limit_max + 0.1 * limit_diff
        limit_min = limit_min - 0.1 * limit_diff
        limit = (limit_min, limit_max)

    plt.close('all')
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(4, 4))
    get_r_vs_ff_scatter_inner(
        ax=ax, perf_ff=perf_ff, perf_r_main=perf_r_main, xlabel=None, ylabel=None, limit=limit,
        prefix='all # of iterations and readout\n', show_diff_hist=show_diff_hist,
    )
    fig.subplots_adjust(left=0.125, right=0.99, bottom=0.125, top=0.99)
    # https://stackoverflow.com/a/26892326

    suptitle = f'scatter_r_vs_ff_{metric}' + suptitle_suffix
    # fig.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')
    fig.text(0.5, 0.0, xlabel, ha='center', va='bottom')
    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical', ha='left')
    savefig(fig, key=join(dir_plot, suptitle + '.pdf'))
    plt.show()
    plt.close('all')
    # secondary plot, R vs FF, grouped by readout type or number of iterations
    # this only works when we EXACTLY have 4 readout types and 6 iterations.
    fig, axes = plt.subplots(2, 5, squeeze=False, figsize=(10, 4), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, (key, level) in enumerate(chain(
            zip_longest(perf_r.index.get_level_values('rcnn_bl_cls').unique(),
                        ('rcnn_bl_cls',), fillvalue='rcnn_bl_cls'),
            zip_longest(perf_r.index.get_level_values('readout_type').unique(),
                        ('readout_type',), fillvalue='readout_type'),
    )):
        # print((key, level))
        # get that r.
        other_level = tuple(set(total_levels) - {level})
        assert len(other_level) == 1
        perf_r_main = perf_r.xs(key, level=level).unstack(other_level).max(axis=1).to_frame(name='perf_r')
        ax = axes[idx]
        if level == 'readout_type':
            key_to_disp = readout_type_mapping[key]
        else:
            key_to_disp = key
        get_r_vs_ff_scatter_inner(
            ax=ax, perf_ff=perf_ff, perf_r_main=perf_r_main,
            xlabel=None,
            ylabel=None,
            limit=limit,
            prefix={'rcnn_bl_cls': '# of iterations', 'readout_type': 'readout'}[level] + f' = {key_to_disp}' + '\n',
            # remove_x_axis_labels=not (idx >= 5),
            # remove_y_axis_labels=not (idx == 5),
            show_diff_hist=show_diff_hist,
        )
    suptitle = f'scatter_r_vs_ff_{metric}_2nd' + suptitle_suffix
    # fig.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')
    fig.text(0.5, 0.0, xlabel, ha='center', va='bottom')
    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical', ha='left')
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.99, wspace=0.1, hspace=0.1)
    savefig(fig, key=join(dir_plot, suptitle + '.pdf'))
    plt.show()
    # third plot, R vs FF, for every combination of readout type or number of iterations

    plt.close('all')
    # secondary plot, R vs FF, grouped by readout type or number of iterations
    # this only works when we EXACTLY have 4 readout types and 6 iterations.
    fig, axes = plt.subplots(4, 6, squeeze=False, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    index_num_cls = perf_r.index.get_level_values('rcnn_bl_cls').values
    index_readout_type = perf_r.index.get_level_values('readout_type').values

    index_readout_cls = np.asarray(
        [index_readout_type, index_num_cls]
    ).T.tolist()
    index_readout_cls = [tuple(x) for x in index_readout_cls]
    index_readout_cls_good_order = list(product(readout_type_order, range(2, 7 + 1)))
    index_readout_cls = sorted(set(index_readout_cls))
    assert set(index_readout_cls_good_order) == set(index_readout_cls)
    assert len(index_readout_cls) <= 24
    for idx, key_this in enumerate(index_readout_cls_good_order):
        perf_r_main = perf_r.xs(
            key_this, level=('readout_type', 'rcnn_bl_cls')
        ).to_frame(name='perf_r')
        ax = axes[idx]

        idx_row, idx_col = idx // 6, idx % 6

        get_r_vs_ff_scatter_inner(
            ax=ax, perf_ff=perf_ff, perf_r_main=perf_r_main,
            xlabel=None,
            ylabel=None if idx_col != 0 else readout_type_mapping[key_this[0]],
            limit=limit,
            prefix='',
            show_diff_hist=show_diff_hist,
            title=None if idx_row != 0 else f'{key_this[1]} iterations',
            legend=(idx == 0),
            show_text=(idx == 0),
        )

    suptitle = f'scatter_r_vs_ff_{metric}_3rd' + suptitle_suffix
    # fig.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')
    fig.text(0.5, 0.0, xlabel, ha='center', va='bottom')
    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical', ha='left')
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.1)
    savefig(fig, key=join(dir_plot, suptitle + '.pdf'))
    plt.show()


def get_hyperparameter_effect_result(
        *, df_in, metric_list, max_cls,
        num_channels_to_iterate_allowlist,
        num_layers_to_iterate_allowlist,
):
    ret = dict()
    for metric in metric_list:
        print(metric)
        assert metric in df_in.columns
        df_this = df_in.loc[:, [metric, 'num_param']].rename(columns={metric: 'perf'})
        columns, df_ff, df_r = preprocess(
            df_this, max_cls=max_cls,
            axes_to_reduce=['model_seed']
        )
        # filter out results.
        df_ff = df_ff[df_ff.index.get_level_values('out_channel').isin(num_channels_to_iterate_allowlist)]
        df_ff = df_ff[df_ff.index.get_level_values('num_layer').isin(num_layers_to_iterate_allowlist)]

        df_r = df_r[df_r.index.get_level_values('out_channel').isin(num_channels_to_iterate_allowlist)]
        df_r = df_r[df_r.index.get_level_values('num_layer').isin(num_layers_to_iterate_allowlist)]
        ret[metric] = {
            'df_ff': df_ff.sort_index(),
            'df_r': df_r.sort_index(),
        }

    return ret


def get_perf_vs_param_result(
        *, df_in, metric_list, max_cls,
        num_channels_to_iterate_allowlist,
        num_layers_to_iterate_allowlist
):
    tbl_data = dict()
    for metric in metric_list:
        print(metric)
        df_this_metric = []
        assert metric in df_in.columns
        df_this = df_in.loc[:, [metric, 'num_param']].rename(columns={metric: 'perf'})
        columns, df_ff, df_r = preprocess(
            df_this, max_cls=max_cls,
            axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed']
        )
        # by earlier eyeballing, I find that
        # models with more channel most of the time always have bigger size than models with fewer channels
        # regardless of number of layers.
        # therefore, it's good to iterate over channels after iterating over layers.
        train_keep_to_check = df_ff.index.get_level_values('train_keep').unique()
        num_ch_to_check = df_ff.index.get_level_values('out_channel').unique()
        num_layer_to_check = df_ff.index.get_level_values('num_layer').unique()

        for (train_keep, num_ch, num_layer) in product(
                train_keep_to_check, num_ch_to_check, num_layer_to_check
        ):
            if num_ch not in num_channels_to_iterate_allowlist or num_layer not in num_layers_to_iterate_allowlist:
                continue
            # get data
            df_ff_this = df_ff.xs((train_keep, num_ch, num_layer), level=('train_keep', 'out_channel', 'num_layer'))
            df_r_this = df_r.xs((train_keep, num_ch, num_layer), level=('train_keep', 'out_channel', 'num_layer'))
            # print((train_keep, num_ch, num_layer))
            # print(df_ff_this.shape, df_r_this.shape)
            assert df_ff_this.shape == (1, 4)
            assert df_r_this.shape == (24, 4)
            readout_type_to_use = df_r_this.index.get_level_values('readout_type').unique()
            assert set(readout_type_to_use) == set(readout_type_order)
            series_ff = df_ff_this.iloc[0]
            for readout_type_idx, readout_type in enumerate(readout_type_order):
                df_r_this_readout = df_r_this.xs(readout_type, level='readout_type')
                perf_max = df_r_this_readout['perf_mean'].max()
                df_this_metric.append(
                    {
                        'train_keep': train_keep,
                        'out_channel': num_ch,
                        'num_layer': num_layer,
                        'num_param_in_k': round(series_ff['num_param_mean'] / 1000),
                        'model_size_name': '{}ch,{}Rl'.format(num_ch, (num_layer - 1) // 2),
                        'perf_ff': series_ff['perf_mean'],
                        # hack to preserve order
                        'readout_type': str(readout_type_idx) + '.' + readout_type_mapping[readout_type],
                        'perf_r': perf_max,
                        'improvement_perc': (perf_max - series_ff['perf_mean'])/series_ff['perf_mean'] * 100
                    }
                )
                df_this_metric[-1]['col_name'] = '{} ({}K)'.format(
                    df_this_metric[-1]['model_size_name'],
                    df_this_metric[-1]['num_param_in_k']
                )
        df_this_metric = pd.DataFrame(
            df_this_metric, columns=[
                'train_keep', 'out_channel', 'num_layer', 'num_param_in_k',
                'perf_ff', 'readout_type', 'perf_r', 'improvement_perc', 'model_size_name', 'col_name'
            ]
        ).set_index(keys=['readout_type', 'train_keep', 'out_channel', 'num_layer'], verify_integrity=True)
        tbl_data[metric] = df_this_metric
    return tbl_data


def main_loop_for_additional_tables(
        *, df_in, dir_key, metric_list=None, display=None, max_cls=7,
        num_channels_to_iterate_allowlist=(8, 16, 32, 48, 64),
        num_layers_to_iterate_allowlist=(3, 5),
):
    if display is None:
        def display(_):
            pass
    if metric_list is None:
        metric_list = [x for x in ['cc2_normed_avg', 'cc2_raw_avg', 'cc_raw_avg'] if x in df_in.columns]
    # perf vs param
    perf_vs_param_data = get_perf_vs_param_result(
        df_in=df_in, metric_list=metric_list, max_cls=max_cls,
        num_channels_to_iterate_allowlist=num_channels_to_iterate_allowlist,
        num_layers_to_iterate_allowlist=num_layers_to_iterate_allowlist,
    )

    # another thing, is the use of individual hyperparameters
    hyperameter_data = get_hyperparameter_effect_result(
        df_in=df_in, metric_list=metric_list, max_cls=max_cls,
        num_channels_to_iterate_allowlist=num_channels_to_iterate_allowlist,
        num_layers_to_iterate_allowlist=num_layers_to_iterate_allowlist,
    )

    return {
        'hyperameter_data': hyperameter_data,
        'perf_vs_param_data': perf_vs_param_data,
    }


def main_loop(df_in, dir_key, metric_list=None, display=None, max_cls=7,
              check_no_missing_data=True):
    if display is None:
        def display(_):
            pass
    if metric_list is None:
        metric_list = [x for x in ['cc2_normed_avg', 'cc2_raw_avg', 'cc_raw_avg'] if x in df_in.columns]
    tbl_data = dict()
    for metric in metric_list:
        print(metric)
        assert metric in df_in.columns
        df_this = df_in.loc[:, [metric, 'num_param']].rename(columns={metric: 'perf'})

        get_r_vs_ff_scatter(
            df_this, max_cls=max_cls,
            axes_to_reduce=['model_seed'],
            metric=metric,
            dir_plot=dir_key,
        )

        get_r_vs_ff_scatter(
            df_this, max_cls=max_cls,
            axes_to_reduce=['model_seed'],
            metric=metric,
            dir_plot=dir_key,
            show_diff_hist=True,
        )

        # not very useful.
        # for deep_l in [[4], [5], [6], [4, 5, 6]]:
        #     get_r_vs_ff_scatter(
        #         df_this, max_cls=max_cls,
        #         axes_to_reduce=['model_seed'],
        #         metric=metric,
        #         dir_plot=dir_key,
        #         deeper_ff=deep_l,
        #     )
        #
        #     get_r_vs_ff_scatter(
        #         df_this, max_cls=max_cls,
        #         axes_to_reduce=['model_seed'],
        #         metric=metric,
        #         dir_plot=dir_key,
        #         deeper_ff=deep_l,
        #         show_diff_hist=True,
        #     )

        get_perf_over_cls_data(df_this, max_cls=max_cls, display=display,
                               axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed'])

        get_perf_over_cls_data(df_this, max_cls=max_cls, display=display,
                               axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed',
                                               'num_layer', 'out_channel']
                               )

        get_ff_vs_best_r_data(
            df_this,
            axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed'],
            max_cls=max_cls, display=display,
            reference_num_layer_ff=3,
            override_ff_num_layer=[2, 3, 4, 5, 6]
        )

        tbl_data[metric] = loop_over_train_size(df_this, metric=metric, dir_plot=dir_key,
                                                display=display, max_cls=max_cls,
                                                check_no_missing_data=check_no_missing_data)

    # TODO: collect FF data as well
    # combine ALL tbl_data, and then save a huge csv.

    data_all = []
    keys_all = []

    # return {
    #         'title': title,
    #         'num_param_df': num_param_df,
    #         'num_param_df_diff': (num_param_df / num_param_df.loc[1] - 1),
    #         'perf_df': perf_df,
    #         'perf_df_diff': (perf_df / perf_df.loc[1] - 1),
    #     }
    data_all_ff = []
    data_all_ff_t = []
    keys_all_ff = []

    for metric_key, metric_data in tbl_data.items():
        for train_size_key, train_size_data in metric_data.items():
            data_non_ff = train_size_data['non-ff']
            for data_one in data_non_ff:
                data_all.extend(
                    [
                        data_one['num_param_df'],
                        data_one['num_param_df_diff'] * 100,
                        data_one['perf_df'],
                        data_one['perf_df_diff'] * 100,
                    ]
                )
                keys_all.extend(
                    [
                        (metric_key, train_size_key, data_one['title'], 'num_param_df'),
                        (metric_key, train_size_key, data_one['title'], 'num_param_df_diff_%'),
                        (metric_key, train_size_key, data_one['title'], 'perf_df'),
                        (metric_key, train_size_key, data_one['title'], 'perf_df_diff_%'),
                    ]
                )
            data_ff = train_size_data['ff']
            data_all_ff.extend(
                [
                    data_ff['num_param_mean'],
                    data_ff['num_param_mean_diff'] * 100,
                    data_ff['perf_mean'],
                    data_ff['perf_mean_diff'] * 100,
                ]
            )
            data_all_ff_t.extend(
                [
                    data_ff['num_param_mean_t'],
                    data_ff['num_param_mean_t_diff'] * 100,
                    data_ff['perf_mean_t'],
                    data_ff['perf_mean_t_diff'] * 100,
                ]
            )
            keys_all_ff.extend(
                [
                    (metric_key, train_size_key, 'num_param_mean'),
                    (metric_key, train_size_key, 'num_param_mean_diff_%'),
                    (metric_key, train_size_key, 'perf_mean'),
                    (metric_key, train_size_key, 'perf_mean_diff_%'),
                ]
            )

    tbl_data_all: pd.DataFrame = pd.concat(data_all, axis=0,
                                           keys=keys_all,
                                           verify_integrity=True,
                                           names=['metric', 'train_size', 'case', 'subframe'])

    csv_f = join(dir_dict['analyses'], dir_key, 'main_results_aggregated.csv')
    makedirs(dirname(csv_f), exist_ok=True)
    tbl_data_all.to_csv(
        csv_f
    )

    tbl_data_all_ff: pd.DataFrame = pd.concat(data_all_ff, axis=0,
                                              keys=keys_all_ff,
                                              verify_integrity=True,
                                              names=['metric', 'train_size', 'subframe'])
    csv_f_ff = join(dir_dict['analyses'], dir_key, 'main_results_ff.csv')
    tbl_data_all_ff.to_csv(
        csv_f_ff
    )

    tbl_data_all_ff_t: pd.DataFrame = pd.concat(data_all_ff_t, axis=0,
                                                keys=keys_all_ff,
                                                verify_integrity=True,
                                                names=['metric', 'train_size', 'subframe'])
    csv_f_ff_t = join(dir_dict['analyses'], dir_key, 'main_results_ff_t.csv')
    tbl_data_all_ff_t.to_csv(
        csv_f_ff_t
    )


def loop_over_train_size(df_in, *, metric, dir_plot, display, max_cls, check_no_missing_data):
    tbl_data = dict()
    for train_keep in df_in.index.get_level_values('train_keep').unique():
        print(train_keep)
        readout_types_to_handle = sorted(
            list(set(df_in.index.get_level_values('readout_type').unique()) - {'legacy'}),
            # simpler models (inst-, -last) first,
            # complicated models (cm-, -avg) later.
            key=lambda x: readout_type_order.index(x)
        )
        print(readout_types_to_handle)
        df_this = df_in.xs(train_keep, level='train_keep').sort_index()
        plt.close('all')
        # close('all'), show() pair is needed to save memory. otherwise, there are too many figures.
        # I don't like using this pair inside the code because I feel this is a global state changing stuff.
        # ideally, they should be called in script instead of a library function.
        tbl_data_this = process_one_case(
            df_this, metric=metric, train_keep=train_keep,
            readout_types_to_handle=readout_types_to_handle,
            max_cls=max_cls,
            check_no_missing_data=check_no_missing_data,
            dir_plot=dir_plot,
            display=display,
        )
        plt.show()
        tbl_data[train_keep] = tbl_data_this
    return tbl_data


def check_model_seeds(df_in):
    assert set(df_in.index.get_level_values('model_seed').unique()) == {0, 1}
    data_0 = df_in['perf'].xs(0, level='model_seed').sort_index()
    data_1 = df_in['perf'].xs(1, level='model_seed').sort_index()
    assert data_0.index.equals(data_1.index)
    data_0_raw = data_0.values
    data_1_raw = data_1.values

    print(f'seed=0, mean {data_0_raw.mean()} std {data_0_raw.std()}')
    print(f'seed=1, mean {data_1_raw.mean()} std {data_1_raw.std()}')
    print('corr', pearsonr(data_0_raw, data_1_raw)[0])

    # check that num_param are the same.
    data_0_num_param = df_in['num_param'].xs(0, level='model_seed').sort_index()
    data_1_num_param = df_in['num_param'].xs(1, level='model_seed').sort_index()
    assert data_0_num_param.equals(data_1_num_param)
    return 2


def avg_out_seed(df_in):
    df_perf = df_in['perf'].unstack('model_seed').mean(axis=1).sort_index()
    df_num_param = df_in['num_param'].xs(0, level='model_seed').sort_index()
    df_perf.name = 'perf'
    df_num_param.name = 'num_param'
    assert df_perf.index.equals(df_num_param.index)
    ret = pd.concat([df_perf, df_num_param], axis='columns')
    assert ret.index.equals(df_perf.index)
    assert ret.index.equals(df_num_param.index)
    return ret


def process_one_case(df_in, *, metric, train_keep,
                     readout_types_to_handle,
                     max_cls=None,
                     check_no_missing_data,
                     dir_plot,
                     display
                     ):
    print(df_in.shape)
    # for each metric.

    # 1. compare seed=0 and seed=1. make sure things are ok.
    # the larger the training size is, the more stable across seeds.
    check_model_seeds(df_in)

    # # 2. take average of model seeds.
    # df_in = avg_out_seed(df_in)
    # print(df_in.shape)
    #
    # # 3. for each combination (ff, PROPER cm-avg x num_layer, PROPER cm-last x num_layer) x (out_channel, num_layer)
    # #    compute average. make sure each one has SAME number of settings (handle cm-avg/cm-last ambiguity for ff)
    # data_ff = process_ff_models(df_in)
    #
    # data_r_list = [
    #     process_recurrent_models(df_in, x) for x in readout_types_to_handle
    # ]

    #     data_r_cm_avg = process_recurrent_models(df_in, 'cm-avg')
    #     data_r_cm_last = process_recurrent_models(df_in, 'cm-last')

    #     data_r_inst_avg = process_recurrent_models(df_in, 'inst-avg')
    #     data_r_inst_last = process_recurrent_models(df_in, 'inst-last')
    # 5. plot/table! maybe have both combined / separate results.

    return plot_one_case(
        df_in=df_in,
        r_name_list=readout_types_to_handle,
        max_cls=max_cls,
        suptitle=f'train size={train_keep}, {metric}',
        ylabel=metric_dict[metric],
        check_no_missing_data=check_no_missing_data,
        dir_plot=dir_plot,
        display=display,
    )


def plot_only_ff(*, ax, data, ylabel, check_no_missing_data, display, num_variant, data_ff_per_layer_original):
    # only show data with >=8 channels. if needed by reviewers, will do a separate plot.
    # >=8 channel data is enough to illustrate my point, and data for <8 channels are not complete.
    data = data.iloc[data.index.get_level_values('out_channel') >= 8, :].sort_index()

    perf_mean = data['perf_mean']
    perf_sem = data['perf_sem']

    num_param_mean = data['num_param_mean']
    num_param_sem = data['num_param_sem']

    perf_mean = perf_mean.unstack('num_layer')
    perf_sem = perf_sem.unstack('num_layer')

    num_param_mean = num_param_mean.unstack('num_layer')
    num_param_sem = num_param_sem.unstack('num_layer')

    if check_no_missing_data:
        assert np.all(np.isfinite(perf_mean.values))
        assert np.all(np.isfinite(perf_sem.values))
        assert np.all(np.isfinite(num_param_mean.values))
        assert np.all(np.isfinite(num_param_sem.values))

    perf_min = np.nanmin(perf_mean.values)
    perf_max = np.nanmax(perf_mean.values)
    margin = (perf_max - perf_min) * 0.05
    #     perf = perf.mean(axis=0)

    #     print(num_param)
    #     print(perf)

    # not good for my usage
    # perf_mean.plot(ax=ax, kind='bar', yerr=perf_sem, ylim=(perf_min - margin, perf_max + 4 * margin), rot=0)
    # ax.set_title('FF models')
    # ax.set_ylabel(ylabel)
    # ax.set_xlabel('# of channels')
    #
    # ax.legend(loc='upper left', ncol=perf_mean.shape[1], bbox_to_anchor=(0.01, 0.99),
    #           borderaxespad=0., fontsize='x-small', handletextpad=0, title='# of layers',
    #           )

    perf_mean.T.plot(ax=ax, kind='bar', yerr=perf_sem.T, ylim=(perf_min - margin, perf_max + 4 * margin), rot=0)
    ax.set_title(f'FF models, n={num_variant} per bar')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('# of layers')

    # plot extra bars using data_ff_per_layer_original
    perf_mean_per_layer = data_ff_per_layer_original['perf_mean']
    assert perf_mean_per_layer.equals(perf_mean_per_layer.sort_index())

    # ax.errorbar(...
    ax.plot(
        np.arange(perf_mean_per_layer.size),
        perf_mean_per_layer,
        # yerr=data_ff_per_layer_original['perf_sem'],
        color='k',
        marker='x',
    )

    ax.legend(loc='upper left', ncol=perf_mean.T.shape[1], bbox_to_anchor=(0.01, 0.99),
              borderaxespad=0., fontsize='x-small', handletextpad=0, title='# of channels',
              )
    # hack for my final plot
    assert perf_mean_per_layer.index.values.tolist() == [2,3,4,5,6]
    ax.set_xticklabels(
        ['2', '3 (=1R)', '4', '5 (=2R)', '6']
    )

    # plot extra data

    # perf_mean_per_layer.plot(ax=ax, kind='line')

    #     print(dir(legend))
    #     print(legend, 'aaaa')
    #     raise RuntimeError

    # use num_layer = 3, num_channel = 16 as base line
    display(num_param_mean)
    num_param_mean_diff = (num_param_mean / num_param_mean.loc[16] - 1)
    display(num_param_mean_diff.style.format("{:.3%}"))
    display(perf_mean)
    perf_mean_diff = (perf_mean / perf_mean.loc[16] - 1)
    display(perf_mean_diff.style.format("{:.3%}"))
    num_param_mean_t = num_param_mean.T
    perf_mean_t = perf_mean.T
    display(num_param_mean_t)
    num_param_mean_t_diff = (num_param_mean_t / num_param_mean_t.loc[3] - 1)
    display(num_param_mean_t_diff.style.format("{:.3%}"))
    display(perf_mean_t)
    perf_mean_t_diff = (perf_mean_t / perf_mean_t.loc[3] - 1)
    display(perf_mean_t_diff.style.format("{:.3%}"))

    return {
        'num_param_mean': num_param_mean,
        'num_param_mean_diff': num_param_mean_diff,
        'perf_mean': perf_mean,
        'perf_mean_diff': perf_mean_diff,
        'num_param_mean_t': num_param_mean_t,
        'num_param_mean_t_diff': num_param_mean_t_diff,
        'perf_mean_t': perf_mean_t,
        'perf_mean_t_diff': perf_mean_t_diff,
    }


def get_data_helper(*, df_in, r_name_list, max_cls, axes_to_reduce):
    _, data_ff, df_r_single, num_variant = preprocess(
        df_in, max_cls=max_cls, axes_to_reduce=axes_to_reduce,
        return_n=True
    )
    data_r_list = [df_r_single.xs(x, level='readout_type').sort_index() for x in r_name_list]
    # check index.
    if isinstance(data_ff, pd.DataFrame):
        index_reference_r: pd.MultiIndex = data_r_list[0].index
        index_reference_ff = index_reference_r.get_loc_level(2, level='rcnn_bl_cls')[1]
        for cls_this in index_reference_r.get_level_values('rcnn_bl_cls').unique():
            assert index_reference_ff.equals(index_reference_r.get_loc_level(cls_this, level='rcnn_bl_cls')[1])
        for data_r_this in data_r_list:
            assert data_r_this.index.equals(index_reference_r)
        assert index_reference_ff.isin(data_ff.index).all()
        assert index_reference_ff.names == data_ff.index.names
        # data_ff = data_ff.loc[index_reference_ff].sort_index()
        return data_ff, data_r_list, num_variant, index_reference_ff
    else:
        return data_ff, data_r_list, num_variant, None


def plot_one_case(
        *,
        df_in,
        r_name_list,
        max_cls=None,
        suptitle=None,
        ylabel,
        check_no_missing_data,
        dir_plot,
        display,
):
    # get per num_layer, out_channel data
    data_ff, data_r_list, num_variant, index_reference = get_data_helper(
        df_in=df_in, r_name_list=r_name_list, max_cls=max_cls,
        axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type', 'model_seed']
    )
    # get overall data

    data_ff_overall, data_r_list_overall, num_variant_overall, _ = get_data_helper(
        df_in=df_in, r_name_list=r_name_list, max_cls=max_cls,
        axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type',
                        'model_seed', 'num_layer', 'out_channel']
    )
    # get per num_layer data
    data_ff_per_layer, data_r_list_per_layer, num_variant_per_layer, index_reference_per_layer = get_data_helper(
        df_in=df_in, r_name_list=r_name_list, max_cls=max_cls,
        axes_to_reduce=['act_fn', 'ff_1st_bn_before_act', 'loss_type',
                        'model_seed', 'out_channel']
    )
    data_ff_per_layer_original = data_ff_per_layer
    data_ff_per_layer = data_ff_per_layer.loc[index_reference_per_layer].sort_index()

    index_out_channel = index_reference.get_level_values('out_channel').values
    index_num_layer = index_reference.get_level_values('num_layer').values

    unique_channel_layer = sorted(set(zip(index_out_channel.tolist(), index_num_layer.tolist())))

    num_setup = len(unique_channel_layer)
    nrows = (num_setup - 1) // 2 + 1
    ncols = 2

    assert len(r_name_list) == len(data_r_list)

    # assert nrows == 3
    if nrows == 3:
        fig_h = 8
    else:
        fig_h = 8 / 3 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, fig_h), squeeze=False, dpi=300)
    fig.subplots_adjust(left=0.1, right=0.975, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    assert suptitle is not None
    #     if suptitle is not None:
    #         fig.suptitle(f'{suptitle}')
    axes = axes.ravel()

    tbl_data_all = []

    for idx, setup_this in enumerate(unique_channel_layer):
        ax = axes[idx]
        tbl_data_all.append(plot_one_case_inner(
            ax=ax,
            data_ff=data_ff.loc[index_reference].sort_index().xs(setup_this, level=('out_channel', 'num_layer')),
            data_r_list=[x.xs(setup_this, level=('out_channel', 'num_layer')) for x in data_r_list],
            setup=setup_this,
            max_cls=max_cls,
            r_name_list=r_name_list,
            num_variant=num_variant,
            title_override=None,
            ylabel=None,
            xlabel=None,
            check_no_missing_data=check_no_missing_data,
            xticklabels_off=False if idx // 2 == nrows - 1 else True,
            display=display,
        ))
    fig.text(0.5, 0.0, '# of iterations', ha='center', va='bottom')
    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical', ha='left')
    # fig.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')

    savefig(fig, join(dir_plot, suptitle.replace(' ', '+') + 'detailed.pdf'))
    fig_main, axes_main = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), squeeze=False, dpi=300)
    fig_main.subplots_adjust(left=0.2, right=0.975, hspace=0.2)
    axes_main = axes_main.ravel()
    # collect everything together.
    tbl_data_all.append(plot_one_case_inner(
        ax=axes_main[0],
        data_ff=data_ff_overall,
        data_r_list=data_r_list_overall,
        setup=None,
        title_override=f'All, n={num_variant_overall}',
        max_cls=max_cls,
        r_name_list=r_name_list,
        num_variant=num_variant_overall,
        ylabel=ylabel,
        xlabel='# of iterations',
        check_no_missing_data=check_no_missing_data,
        xticklabels_off=False,
        display=display,
    ))
    savefig(fig_main, join(dir_plot, suptitle.replace(' ', '+') + 'main.pdf'))

    # only FF.
    # for FF, let's ignore saving to CSV, because the format is a bit different,
    # and it's not a huge amount of work anyway.
    # maybe I can do it later.

    fig_ff, axes_ff = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), squeeze=False, dpi=300)
    fig_ff.subplots_adjust(left=0.2, right=0.975, hspace=0.2)
    axes_ff = axes_ff.ravel()
    tbl_data_ff = plot_only_ff(
        ax=axes_ff[0],
        # do not filter.
        data=data_ff,
        ylabel=ylabel,
        check_no_missing_data=check_no_missing_data,
        display=display,
        num_variant=num_variant,
        data_ff_per_layer_original=data_ff_per_layer_original,
    )
    # not to be used.
    savefig(fig_ff, join(dir_plot, suptitle.replace(' ', '+') + 'ff.pdf'))
    # fig_main.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')

    # plot perf vs num_param scatter.
    # for cls = 1, 3, 5, 7, cm-avg. because this is most apparent.
    cm_avg_idx = r_name_list.index('cm-avg')

    data_ff_for_scatter, data_r_list_for_scatter, num_variant_for_scatter, index_reference_scatter = get_data_helper(
        df_in=df_in, r_name_list=r_name_list, max_cls=max_cls,
        axes_to_reduce=['model_seed']
    )
    plot_scatter_plot(
        data_ff=data_ff_for_scatter.loc[index_reference_scatter].sort_index(),
        data_r=data_r_list_for_scatter[cm_avg_idx],
        ylabel=ylabel,
        title=f"model performance vs. model size for different # of iterations, 'cm-avg' readout type",
        suptitle=suptitle,
        num_seed=num_variant_for_scatter,
        dir_plot=dir_plot,
    )

    # per layer
    fig_per_layer, axes_per_layer = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 3.5), squeeze=True,
    )
    fig_per_layer.subplots_adjust(left=0.125, right=0.975, bottom=0.125, top=0.9, wspace=0.3, hspace=0.2)
    for idx_per_layer, num_layer in enumerate(index_reference_per_layer.get_level_values('num_layer').unique()):
        plot_one_case_inner(
            ax=axes_per_layer[idx_per_layer],
            data_ff=data_ff_per_layer.xs(
                # must be None here because it's a single index, not multiindex
                num_layer, level=None
            ),
            data_r_list=[
                x.xs(num_layer, level='num_layer') for x in data_r_list_per_layer
            ],
            setup=(num_layer,),
            max_cls=max_cls,
            r_name_list=r_name_list,
            num_variant=num_variant_per_layer,
            title_override=None,
            ylabel=ylabel,
            xlabel='# of iterations',
            check_no_missing_data=check_no_missing_data,
            xticklabels_off=False,
            display=display,
        )
    # fig_per_layer.text(0.5, 0.0, '# of iterations', ha='center', va='bottom')
    # fig_per_layer.text(0.0, 0.5, ylabel, va='center', rotation='vertical', ha='left')
    savefig(fig_per_layer, join(dir_plot, suptitle.replace(' ', '+') + '_per_layer.pdf'))

    return {
        'non-ff': tbl_data_all,
        'ff': tbl_data_ff,
    }


def plot_scatter_plot_inner_mean(*,
                                 ax_scatter,
                                 data,
                                 color,
                                 ):
    # get unique combinations of num layer and num channel
    index_out_channel = data.index.get_level_values('out_channel').values
    index_num_layer = data.index.get_level_values('num_layer').values

    data_channel_layer = np.asarray([index_out_channel, index_num_layer]).T
    #     print(data_channel_layer.dtype, data_channel_layer.shape)
    unique_channel_layer = np.unique(data_channel_layer, axis=0).tolist()
    # assert len(unique_channel_layer) == 6  # two layer configs x three channel configs

    for key_this in unique_channel_layer:
        key_this = tuple(key_this)
        (c_this, l_this) = key_this
        df_this = data.xs(key=(c_this, l_this), level=('out_channel', 'num_layer'))
        perf_mean = df_this['perf_mean'].mean()
        num_param_mean = df_this['num_param_mean'].mean()
        ax_scatter.plot([num_param_mean - 500, num_param_mean + 500], [perf_mean, perf_mean], color=color,
                        linestyle='--')


def plot_scatter_plot(*, data_ff, data_r, title, ylabel, num_seed, dir_plot, suptitle):
    fig_scatter, ax_scatter = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), squeeze=True, dpi=300)
    fig_scatter.subplots_adjust(left=0.1, right=0.975, bottom=0.1, top=0.9)
    ax_scatter.set_title(title)

    cls_to_show_r = [3, 5, 7]
    color_to_show_r = ['r', 'g', 'b']
    assert len(cls_to_show_r) == len(color_to_show_r)
    num_variant = data_ff.shape[0] * num_seed
    ax_scatter.scatter(data_ff['num_param_mean'], data_ff['perf_mean'], color='k', s=6, label='1')
    ymin, ymax = data_ff['perf_mean'].min(), data_ff['perf_mean'].max()
    plot_scatter_plot_inner_mean(ax_scatter=ax_scatter, data=data_ff, color='k')

    for cls_this, color in zip(cls_to_show_r, color_to_show_r):
        data_r_this = data_r.xs(cls_this, level='rcnn_bl_cls')
        assert num_variant == data_r_this.shape[0] * num_seed
        ax_scatter.scatter(data_r_this['num_param_mean'], data_r_this['perf_mean'], color=color, s=6,
                           label=str(cls_this))
        ymin, ymax = min(ymin, data_r_this['perf_mean'].min()), max(ymax, data_r_this['perf_mean'].max())
        plot_scatter_plot_inner_mean(ax_scatter=ax_scatter, data=data_r_this, color=color)

    margin = (ymax - ymin) * 0.05
    ax_scatter.set_ylim((ymin - margin, ymax + margin))

    ax_scatter.legend(loc='upper left', ncol=len(cls_to_show_r) + 1, bbox_to_anchor=(0.01, 0.99),
                      borderaxespad=0., fontsize='x-small', handletextpad=0, title='# of iterations',
                      )
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.set_xlabel('# of parameters')
    print(f'scatter plot, {num_variant} variants per iteration')
    fig_scatter.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')
    savefig(fig_scatter, join(dir_plot, suptitle.replace(' ', '+') + 'scatter.pdf'))


def plot_one_case_inner(
        *,
        ax,
        data_ff: pd.DataFrame,
        data_r_list: List[pd.DataFrame],
        setup,
        max_cls,
        r_name_list,
        num_variant,
        title_override,
        ylabel,
        xlabel,
        check_no_missing_data,
        xticklabels_off,
        display,
):
    # remap data_r_list's num layer to be compatible with ff
    # otherwise, when I do `.unstack('rcnn_bl_cls')` later,
    # I won't get a nice N x 7 table with all entries filled.

    #     raise RuntimeError

    #     print(data_r.columns)
    #     print(data_ff.columns)
    if isinstance(data_ff, pd.DataFrame):
        assert data_ff.shape[0] == 1
        data_ff = data_ff.iloc[0]

    assert isinstance(data_ff, pd.Series)
    data_ff = data_ff.to_frame().T
    data_ff = data_ff.copy(deep=True)
    data_ff['rcnn_bl_cls'] = 1
    # no append.
    data_ff = data_ff.set_index('rcnn_bl_cls')
    for data_r in data_r_list:
        assert data_r.shape[0] == data_r.index.get_level_values('rcnn_bl_cls').unique().size
        assert data_r.index.name == 'rcnn_bl_cls'
        assert data_r.index.names == ['rcnn_bl_cls']

    if max_cls is not None:
        data_r_list = [x.iloc[x.index.get_level_values('rcnn_bl_cls') <= max_cls].sort_index() for x in data_r_list]
    augmented_data_r_list = [
        pd.concat(
            [data_ff, x],
            axis=0
        ) for x in data_r_list
    ]
    assert len(r_name_list) == len(data_r_list)
    assert len(r_name_list) == len(augmented_data_r_list)

    def get_one_small_df(col):
        # rcnn_bl_cls rows
        # len(data_r_list) columns
        data_inner = [
            x[col].rename(y) for x, y in zip(
                augmented_data_r_list,
                r_name_list
            )
        ]
        return pd.concat(
            data_inner, axis=1
        )

    num_param_df: pd.DataFrame = get_one_small_df('num_param_mean')
    num_param_sem_df: pd.DataFrame = get_one_small_df('num_param_sem')
    perf_df: pd.DataFrame = get_one_small_df('perf_mean')
    perf_sem_df: pd.DataFrame = get_one_small_df('perf_sem')

    if check_no_missing_data:
        assert np.all(np.isfinite(perf_sem_df.values))
        assert np.all(np.isfinite(perf_df.values))
        assert np.all(np.isfinite(num_param_df.values))
        assert np.all(np.isfinite(num_param_sem_df.values))

    #     display(num_param.mean(axis=0).to_frame().T)
    # #     display(perf)
    #     assert np.all(np.isfinite(num_param.values))
    #     assert np.all(np.isfinite(perf.values))
    perf_min = np.nanmin(perf_df.values)
    perf_max = np.nanmax(perf_df.values)
    margin = (perf_max - perf_min) * 0.05
    #     perf = perf.mean(axis=0)

    #     print(num_param)
    #     print(perf)
    # first one is ff.
    perf_df.iloc[1:].plot(
        ax=ax, kind='line', yerr=perf_sem_df.iloc[1:],
        ylim=(perf_min - margin, perf_max + 4 * margin),
        xlim=(perf_df.iloc[1:].index.values.min() - 0.1, perf_df.iloc[1:].index.values.max() + 0.1),
        xticks=perf_df.iloc[1:].index.values,
        rot=0
    )
    ax.legend(loc='upper left', ncol=perf_df.shape[1], bbox_to_anchor=(0.01, 0.99),
              borderaxespad=0., fontsize='x-small', handletextpad=0,
              labels=readout_type_order_mapped,
              #               title='readout type',
              )
    ax.axhline(y=perf_df.iloc[0, 0], linestyle='-', color='k')
    ax.axhline(y=perf_df.iloc[0, 0] + perf_sem_df.iloc[0, 0], linestyle='--', color='k')
    ax.axhline(y=perf_df.iloc[0, 0] - perf_sem_df.iloc[0, 0], linestyle='--', color='k')
    if setup is not None:
        assert title_override is None

        if len(setup) == 2:
            num_c = setup[0]
            num_l_ff = setup[1]
            num_l_r = (setup[1] - 1) // 2
            layer_text = 'layer' if num_l_r == 1 else 'layers'
            title = f'{num_c} ch, {num_l_r} R {layer_text}, n={num_variant}'
        elif len(setup) == 1:
            num_l_ff = setup[0]
            num_l_r = (setup[0] - 1) // 2
            layer_text = 'layer' if num_l_r == 1 else 'layers'
            title = f'{num_l_r} R {layer_text}, n={num_variant}'
        else:
            raise RuntimeError
    else:
        assert title_override is not None
        title = title_override
    display(num_param_df)
    num_param_df_diff = (num_param_df / num_param_df.loc[1] - 1)
    display(num_param_df_diff.style.format("{:.3%}"))
    display(perf_df)
    perf_df_diff = (perf_df / perf_df.loc[1] - 1)
    display(perf_df_diff.style.format("{:.3%}"))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # xticklabels = ax.get_xticklabels()
    # # first one should be replaced from 1 to 1 (FF)
    # assert xticklabels[0].get_text() == '1'
    # xticklabels[0].set_text('1 (FF)')
    # ax.set_xticklabels(xticklabels)

    if xticklabels_off:
        ax.set_xticklabels([])

    return {
        'title': title,
        'num_param_df': num_param_df,
        'num_param_df_diff': num_param_df_diff,
        'perf_df': perf_df,
        'perf_df_diff': perf_df_diff,
    }
