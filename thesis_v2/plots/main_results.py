from os.path import join, dirname
from os import makedirs
from typing import List

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import savefig
from .. import dir_dict


def main_loop(df_in, dir_key, metric_list=None, display=None, max_cls=7):
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
        tbl_data[metric] = loop_over_train_size(df_this, metric=metric, dir_plot=dir_key,
                                                display=display, max_cls=max_cls)

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


def loop_over_train_size(df_in, *, metric, dir_plot, display, max_cls):
    tbl_data = dict()
    for train_keep in df_in.index.get_level_values('train_keep').unique():
        print(train_keep)
        readout_types_to_handle = sorted(
            list(set(df_in.index.get_level_values('readout_type').unique()) - {'legacy'}),
            # simpler models (inst-, -last) first,
            # complicated models (cm-, -avg) later.
            key=lambda x: ['inst-last', 'cm-last', 'inst-avg', 'cm-avg'].index(x)
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
            check_no_missing_data=True,
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


def process_ff_models(df_in):
    # for each combination of (num_channel, num_layer)
    # generate a sub data frame
    # indexed by ('act_fn', 'ff_1st_bn_before_act', 'loss_type')
    # with columns (perf, num_param)

    data = df_in.xs(1, level='rcnn_bl_cls', drop_level=False)

    #     index_names = data.index.name

    index_out_channel = data.index.get_level_values('out_channel').values
    index_num_layer = data.index.get_level_values('num_layer').values

    data_channel_layer = np.asarray([index_out_channel, index_num_layer]).T
    #     print(data_channel_layer.dtype, data_channel_layer.shape)
    unique_channel_layer = np.unique(data_channel_layer, axis=0).tolist()

    data_dict = dict()

    for key_this in unique_channel_layer:
        key_this = tuple(key_this)
        (c_this, l_this) = key_this
        df_this = data.xs(key=(c_this, l_this), level=('out_channel', 'num_layer'), drop_level=False)
        #         print(df_this.shape)
        # average out over readout_type
        df_this = df_this.unstack('readout_type')
        perf = df_this['perf']
        num_param = df_this['num_param']
        assert perf.shape == num_param.shape
        # num_readout = perf.shape[1]

        #     assert data.shape[1] == 3

        #     print()

        #     for idx, case in data.iterrows():
        #         case_val = case.values
        #         assert case_val.shape == (3,)
        # get non nan values
        #         case_non_nan = case_val[~np.isnan(case)]
        #         assert case_non_nan.size > 0
        #         case_non_nan_debug = np.full_like(case_non_nan, fill_value=case_non_nan[0])
        #         if not np.allclose(case_non_nan, case_non_nan_debug, atol=1e-3):
        #             print(idx, case)
        #             print(case_non_nan, case_non_nan_debug)
        #         assert np.allclose(case_non_nan, case_non_nan_debug, atol=1e-3)

        # actually, probably due to card-to-card variance, difference can appear.
        # I checked one of them.

        # 5120, cc_mean_avg
        # (relu, False, poisson, 2, 32)
        # cm-avg     0.499929
        # cm-last    0.496784
        # legacy          NaN
        # Name: (relu, False, poisson, 2, 32), dtype: float64
        # [0.49992943 0.49678396] [0.49992943 0.49992943]

        # for cm-avg
        # check files
        # models/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/
        # s_selegacy/in_sz50/out_ch32/num_l2/k_l19/k_p3/ptavg/bn_a_fcFalse/actrelu/r_c1/r_psize1/r_ptypeNone/
        # r_acccummean/ff1st_True/ff1stbba_False/rp_none/sc0.01/sm0.000005/lpoisson/m_se0/stats_best.json
        #
        # corr_mean: 0.5004660408667474
        # "best_phase": 2, "best_epoch": 50, "early_stopping_loss": 0.813396155834198
        #
        # and
        #
        # models/yuanyuan_8k_a_3day/maskcnn_polished_with_rcnn_k_bl/
        # s_selegacy/in_sz50/out_ch32/num_l2/k_l19/k_p3/ptavg/bn_a_fcFalse/actrelu/r_c1/r_psize1/r_ptypeNone/
        # r_acccummean/ff1st_True/ff1stbba_False/rp_none/sc0.01/sm0.000005/lpoisson/m_se1/stats_best.json
        #
        # corr_mean: 0.49939281448219086
        # "best_phase": 1, "best_epoch": 1150, "early_stopping_loss": 0.8130963444709778

        # for cm-last
        # check files
        # .... r_acccummean_last/ff1st_True/ff1stbba_False/sc0.01/sm0.000005/lpoisson/m_se0/stats_best.json
        #
        # "corr_mean": 0.49417510516371543
        # "best_phase": 2, "best_epoch": 150, "early_stopping_loss": 0.8139971494674683
        #
        # and
        #
        # .... r_acccummean_last/ff1st_True/ff1stbba_False/sc0.01/sm0.000005/lpoisson/m_se1/stats_best.json
        #
        # corr_mean: 0.49939281448219086
        # {"best_phase": 1, "best_epoch": 1150, "early_stopping_loss": 0.8130963444709778

        # in this case, when seed=1, results are same; when seed=0, they are different.

        # take average to remove card-to-card variance.
        # remove NAs due to incomplete configs.
        # this is very small.

        # print(perf.max(axis=1, skipna=True)-perf.min(axis=1, skipna=True))
        perf = perf.mean(axis=1, skipna=True)
        for _, row_this in num_param.iterrows():
            assert row_this.nunique(dropna=True) == 1

        num_param = num_param.mean(axis=1, skipna=True)
        assert perf.index.equals(num_param.index)

        perf.name = 'perf'
        num_param.name = 'num_param'
        #         print(perf.name, num_param.name)
        ret = pd.concat([perf, num_param], axis='columns')
        #         print(ret.columns)
        #         assert ret.columns == ['perf', 'num_param']
        assert ret.index.equals(perf.index)
        assert ret.index.equals(num_param.index)

        data_dict[key_this] = ret
    return data_dict


def process_recurrent_models(df_in, readout_type):
    data = df_in.xs(readout_type, level='readout_type')
    data = data.iloc[data.index.get_level_values('rcnn_bl_cls') != 1]
    print(data.shape)

    index_out_channel = data.index.get_level_values('out_channel').values
    index_num_layer = data.index.get_level_values('num_layer').values

    data_channel_layer = np.asarray([index_out_channel, index_num_layer]).T
    #     print(data_channel_layer.dtype, data_channel_layer.shape)
    unique_channel_layer = np.unique(data_channel_layer, axis=0).tolist()

    data_dict = dict()

    for key_this in unique_channel_layer:
        key_this = tuple(key_this)
        (c_this, l_this) = key_this
        df_this = data.xs(key=(c_this, l_this), level=('out_channel', 'num_layer'), drop_level=False)
        #         print(df_this.shape)
        # average out over readout_type
        data_dict[key_this] = df_this
    return data_dict


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
    num_seed = check_model_seeds(df_in)

    # 2. take average of model seeds.
    df_in = avg_out_seed(df_in)
    print(df_in.shape)

    # 3. for each combination (ff, PROPER cm-avg x num_layer, PROPER cm-last x num_layer) x (out_channel, num_layer)
    #    compute average. make sure each one has SAME number of settings (handle cm-avg/cm-last ambiguity for ff)
    data_ff = process_ff_models(df_in)

    data_r_list = [
        process_recurrent_models(df_in, x) for x in readout_types_to_handle
    ]

    #     data_r_cm_avg = process_recurrent_models(df_in, 'cm-avg')
    #     data_r_cm_last = process_recurrent_models(df_in, 'cm-last')

    #     data_r_inst_avg = process_recurrent_models(df_in, 'inst-avg')
    #     data_r_inst_last = process_recurrent_models(df_in, 'inst-last')

    recurrent_setups = data_r_list[0].keys()
    for rr in data_r_list:
        assert rr.keys() == recurrent_setups

    # 4. create a mapping between ff (out, num_layer) to similarly sized PROPER recurrents.
    recurrent_to_ff_setup_mapping = dict()
    for setup_r in recurrent_setups:
        recurrent_to_ff_setup_mapping[setup_r] = (setup_r[0], (setup_r[1] - 1) * 2 + 1)
        # we have matching ff models of similar parameters
        assert recurrent_to_ff_setup_mapping[setup_r] in data_ff
    #     print(recurrent_to_ff_setup_mapping)

    # 5. plot/table! maybe have both combined / separate results.

    return plot_one_case(
        data_ff=data_ff,
        data_r_list=data_r_list,
        r_name_list=readout_types_to_handle,
        recurrent_to_ff_setup_mapping=recurrent_to_ff_setup_mapping,
        max_cls=max_cls,
        num_seed=num_seed,
        suptitle=f'train size={train_keep}, {metric}',
        ylabel={
            'cc2_normed_avg': '''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$''',
            'cc2_raw_avg': '''average ${\\mathrm{CC}}^2$''',
            'cc_raw_avg': '''average $\\mathrm{CC}$''',
        }[metric],
        check_no_missing_data=check_no_missing_data,
        dir_plot=dir_plot,
        display=display,
    )


def plot_only_ff(*, ax, data, num_seed, ylabel, check_no_missing_data, display):
    # only show data with >=8 channels. if needed by reviewers, will do a separate plot.
    # >=8 channel data is enough to illustrate my point, and data for <8 channels are not complete.
    data = data.iloc[data.index.get_level_values('out_channel') >= 8, :].sort_index()

    perf = data['perf'].sort_index()
    num_param = data['num_param'].sort_index()
    perf = perf.unstack(['out_channel', 'num_layer'])
    num_variant = num_seed * perf.shape[0]
    assert np.all(np.isfinite(perf.values))
    num_param = num_param.unstack(['out_channel', 'num_layer'])
    assert num_variant == num_seed * num_param.shape[0]
    assert np.all(np.isfinite(num_param.values))

    perf_mean = perf.mean(axis=0)
    perf_sem = perf.std(axis=0, ddof=0) / np.sqrt(num_variant)

    num_param_mean = num_param.mean(axis=0)
    num_param_sem = num_param.std(axis=0, ddof=0) / np.sqrt(num_variant)

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

    perf_mean.plot(ax=ax, kind='bar', yerr=perf_sem, ylim=(perf_min - margin, perf_max + 4 * margin), rot=0)
    ax.set_title('FF models')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('# of channels')

    ax.legend(loc='upper left', ncol=perf_mean.shape[1], bbox_to_anchor=(0.01, 0.99),
              borderaxespad=0., fontsize='x-small', handletextpad=0, title='# of layers',
              )

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


def plot_one_case(
        *,
        data_ff,
        data_r_list,
        r_name_list,
        recurrent_to_ff_setup_mapping,
        max_cls=None,
        num_seed,
        suptitle=None,
        ylabel,
        check_no_missing_data,
        dir_plot,
        display,
):
    # 1 for everything
    for zzz in data_r_list:
        assert len(zzz) == len(data_r_list[0])
        assert zzz.keys() == data_r_list[0].keys()

    num_setup = len(data_r_list[0])
    nrows = (num_setup - 1) // 2 + 1
    ncols = 2

    assert len(r_name_list) == len(data_r_list)

    assert nrows == 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), squeeze=False, dpi=300)
    fig.subplots_adjust(left=0.1, right=0.975, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    assert suptitle is not None
    #     if suptitle is not None:
    #         fig.suptitle(f'{suptitle}')
    axes = axes.ravel()

    tbl_data_all = []

    for idx, setup_this in enumerate(data_r_list[0]):
        ax = axes[idx]
        setup_this_ff = recurrent_to_ff_setup_mapping[setup_this]
        tbl_data_all.append(plot_one_case_inner(
            ax=ax,
            data_ff=data_ff[setup_this_ff],
            data_r_list=[x[setup_this] for x in data_r_list],
            setup_ff=setup_this_ff,
            setup_r=setup_this,
            max_cls=max_cls,
            r_name_list=r_name_list,
            num_seed=num_seed,
            title_override=None,
            ylabel=ylabel if idx % 2 == 0 else None,
            xlabel='# of iterations' if idx // 2 == 2 else None,
            check_no_missing_data=check_no_missing_data,
            xticklabels_off=False if idx // 2 == 2 else True,
            display=display,
        ))
    fig.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')

    savefig(fig, join(dir_plot, suptitle.replace(' ', '+') + 'detailed.pdf'))
    fig_main, axes_main = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5), squeeze=False, dpi=300)
    fig_main.subplots_adjust(left=0.1, right=0.975, hspace=0.2)
    axes_main = axes_main.ravel()
    # collect everything together.
    tbl_data_all.append(plot_one_case_inner(
        ax=axes_main[0],
        data_ff=pd.concat(
            [data_ff[recurrent_to_ff_setup_mapping[s]] for s in data_r_list[0]],
            axis=0,
        ).sort_index(),
        data_r_list=[
            pd.concat(
                [x[s] for s in data_r_list[0]],
                axis=0,
            ).sort_index()
            for x in data_r_list
        ],
        setup_ff=None,
        setup_r=None,
        title_override='FF vs. recurrent models',
        max_cls=max_cls,
        r_name_list=r_name_list,
        num_seed=num_seed,
        ylabel=ylabel,
        xlabel='# of iterations',
        check_no_missing_data=check_no_missing_data,
        xticklabels_off=False,
        display=display,
    ))

    # only FF.
    # for FF, let's ignore saving to CSV, because the format is a bit different,
    # and it's not a huge amount of work anyway.
    # maybe I can do it later.

    tbl_data_ff = plot_only_ff(
        ax=axes_main[1],
        data=pd.concat(
            list(data_ff.values()),
            axis=0,

        ).sort_index(),
        ylabel=None,
        num_seed=num_seed,
        check_no_missing_data=check_no_missing_data,
        display=display,
    )
    fig_main.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')
    savefig(fig_main, join(dir_plot, suptitle.replace(' ', '+') + 'main.pdf'))

    # plot perf vs num_param scatter.
    # for cls = 1, 3, 5, 7, cm-avg. because this is most apparent.
    cm_avg_idx = r_name_list.index('cm-avg')
    plot_scatter_plot(
        data_ff=pd.concat(
            [data_ff[recurrent_to_ff_setup_mapping[s]] for s in data_r_list[cm_avg_idx]],
            axis=0,
        ).sort_index(),
        data_r=pd.concat(
            list(data_r_list[cm_avg_idx].values()),
            axis=0,
        ).sort_index(),
        ylabel=ylabel,
        title=f"model performance vs. model size for different # of iterations, 'cm-avg' readout type",
        suptitle=suptitle,
        num_seed=num_seed,
        dir_plot=dir_plot,
    )

    return {
        'non-ff': tbl_data_all,
        'ff': tbl_data_ff,
    }


def plot_scatter_plot(*, data_ff, data_r, title, ylabel, num_seed, dir_plot, suptitle):
    fig_scatter, ax_scatter = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), squeeze=True, dpi=300)
    fig_scatter.subplots_adjust(left=0.1, right=0.975, bottom=0.1, top=0.9)
    ax_scatter.set_title(title)

    cls_to_show_r = [3, 5, 7]
    color_to_show_r = ['r', 'g', 'b']
    assert len(cls_to_show_r) == len(color_to_show_r)
    num_variant = data_ff.shape[0] * num_seed
    ax_scatter.scatter(data_ff['num_param'], data_ff['perf'], color='k', s=6, label='1')

    ymin, ymax = data_ff['perf'].min(), data_ff['perf'].max()

    for cls_this, color in zip(cls_to_show_r, color_to_show_r):
        data_r_this = data_r.xs(cls_this, level='rcnn_bl_cls')
        assert num_variant == data_r_this.shape[0] * num_seed
        ax_scatter.scatter(data_r_this['num_param'], data_r_this['perf'], color=color, s=6, label=str(cls_this))
        ymin, ymax = min(ymin, data_r_this['perf'].min()), min(ymax, data_r_this['perf'].max())

    margin = (ymax - ymin) * 0.05
    ax_scatter.set_ylim((ymin - margin, ymax + 4 * margin))

    ax_scatter.legend(loc='upper left', ncol=len(cls_to_show_r) + 1, bbox_to_anchor=(0.01, 0.99),
                      borderaxespad=0., fontsize='x-small', handletextpad=0, title='# of iterations',
                      )
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.set_xlabel('# of parameters')
    print(f'scatter plot, {num_variant} variants per iteration')
    fig_scatter.text(0, 1, s=suptitle, horizontalalignment='left', verticalalignment='top')
    savefig(fig_scatter, join(dir_plot, suptitle.replace(' ', '+') + 'scatter.pdf'))


def construct_frame(*, df_list, name_list, num_seed):
    assert len(df_list) == len(name_list)
    series_list = []
    series_sem_list = []
    num_variant = df_list[0].shape[0] * num_seed
    for df, n in zip(df_list, name_list):
        #         display(df)
        assert np.all(np.isfinite(df.values))
        assert df.shape[0] * num_seed == num_variant
        s = df.mean(axis=0)
        s.name = n

        s_sem = df.std(axis=0, ddof=0) / np.sqrt(num_variant)
        s_sem.name = n

        series_list.append(s)
        series_sem_list.append(s_sem)

    return {
        'df_mean': pd.concat(series_list, axis=1).sort_index(),
        'df_sem': pd.concat(series_sem_list, axis=1).sort_index(),
        'num_variant': num_variant,
    }


def plot_one_case_inner(
        *,
        ax,
        data_ff: pd.DataFrame,
        data_r_list: List[pd.DataFrame],
        setup_ff,
        setup_r,
        max_cls,
        r_name_list,
        num_seed,
        title_override,
        ylabel,
        xlabel,
        check_no_missing_data,
        xticklabels_off,
        display,
):
    # remap data_r_list's num layer to be compatible with ff
    data_r_list_new = []
    for data_r_this in data_r_list:
        data_r_this = data_r_this.copy(deep=True)
        num_layer_idx = data_r_this.index.names.index('num_layer')
        data_r_this.index = data_r_this.index.set_levels(
            data_r_this.index.levels[num_layer_idx].map(lambda z: (z - 1) * 2 + 1),
            level=num_layer_idx
        )
        data_r_list_new.append(data_r_this)
    data_r_list = data_r_list_new
    #     raise RuntimeError

    #     print(data_r.columns)
    #     print(data_ff.columns)

    perf_ff = data_ff['perf']
    num_param_ff = data_ff['num_param']

    if max_cls is not None:
        data_r_list = [x.iloc[x.index.get_level_values('rcnn_bl_cls') <= max_cls].sort_index() for x in data_r_list]

    perf_r = [x['perf'] for x in data_r_list]
    num_param_r = [x['num_param'] for x in data_r_list]

    num_param_list = [pd.concat([num_param_ff, x], axis=0).sort_index().unstack('rcnn_bl_cls').sort_index() for x in
                      num_param_r]
    perf_list = [pd.concat([perf_ff, x], axis=0).sort_index().unstack('rcnn_bl_cls').sort_index() for x in perf_r]

    num_param_ret = construct_frame(df_list=num_param_list, name_list=r_name_list, num_seed=num_seed)
    num_param_df: pd.DataFrame = num_param_ret['df_mean']
    num_param_sem_df: pd.DataFrame = num_param_ret['df_sem']
    perf_ret = construct_frame(df_list=perf_list, name_list=r_name_list, num_seed=num_seed)
    perf_df: pd.DataFrame = perf_ret['df_mean']
    perf_sem_df: pd.DataFrame = perf_ret['df_sem']

    if check_no_missing_data:
        assert np.all(np.isfinite(perf_sem_df.values))
        assert np.all(np.isfinite(perf_df.values))
        assert np.all(np.isfinite(num_param_df.values))
        assert np.all(np.isfinite(num_param_sem_df.values))

    num_variant = num_param_ret['num_variant']
    assert num_variant == perf_ret['num_variant']

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

    perf_df.plot(ax=ax, kind='bar', yerr=perf_sem_df, ylim=(perf_min - margin, perf_max + 4 * margin), rot=0)
    ax.legend(loc='upper left', ncol=perf_df.shape[1], bbox_to_anchor=(0.01, 0.99),
              borderaxespad=0., fontsize='x-small', handletextpad=0,
              #               title='readout type',
              )
    if not (setup_ff is None and setup_r is None):
        assert title_override is None
        assert len(setup_ff) == len(setup_r) == 2
        assert setup_ff[0] == setup_r[0]
        num_c = setup_ff[0]
        num_l_ff = setup_ff[1]
        num_l_r = setup_r[1]
        title = f'{num_c} ch, {num_l_ff} C vs. (1 C + {num_l_r - 1} RC), n={num_variant}'

    else:
        assert title_override is not None
        title = f'{title_override}' + f', n={num_variant}'
    print(title)
    display(num_param_df)
    num_param_df_diff = (num_param_df / num_param_df.loc[1] - 1)
    display(num_param_df_diff.style.format("{:.3%}"))
    display(perf_df)
    perf_df_diff = (perf_df / perf_df.loc[1] - 1)
    display(perf_df_diff.style.format("{:.3%}"))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    xticklabels = ax.get_xticklabels()
    # first one should be replaced from 1 to 1 (FF)
    assert xticklabels[0].get_text() == '1'
    xticklabels[0].set_text('1 (FF)')
    ax.set_xticklabels(xticklabels)

    if xticklabels_off:
        ax.set_xticklabels([])

    return {
        'title': title,
        'num_param_df': num_param_df,
        'num_param_df_diff': num_param_df_diff,
        'perf_df': perf_df,
        'perf_df_diff': perf_df_diff,
    }
