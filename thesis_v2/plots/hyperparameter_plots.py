from os.path import join
from os import makedirs
from functools import reduce

from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

from .util import savefig
from .main_results import metric_dict, readout_type_mapping, main_loop_for_additional_tables
from .main_results_tables import process_r, avg_inner


def compare_one_hyperpameter_over_readout_types(
        *, df_merged, hyperparameter_name, column, plot_dir=None, metric,
        hyperparameter_name_friendly=None,
        hy_value_mapping=None,
):
    title = {
        'improvement_abs': '$\\Delta$ of ' + metric_dict[metric],
        'improvement_rel': '%$\\Delta$ of ' + metric_dict[metric]
    }[column]

    for readout_type in df_merged.index.get_level_values('readout_type').unique():
        print(readout_type)
        compare_one_hyperpameter(
            df_merged=df_merged.xs(readout_type, level='readout_type'),
            hyperparameter_name=hyperparameter_name,
            hyperparameter_name_friendly=hyperparameter_name_friendly,
            column=column, plot_dir=plot_dir,
            title=title + f', {readout_type_mapping[readout_type]}',
            plot_name=f'{metric}_{column}_{hyperparameter_name}_{readout_type}',
            hy_value_mapping=hy_value_mapping,
        )


def compare_one_hyperpameter(
        *, df_merged, hyperparameter_name, column, plot_dir=None, plot_name=None, hyperparameter_name_friendly=None,
        title=None, hy_value_mapping=None,
):
    df_this = df_merged[column].unstack(hyperparameter_name)
    assert df_this.shape[1] == 2
    xlabel = df_this.columns[0]
    ylabel = df_this.columns[1]
    assert sorted([xlabel, ylabel]) == [xlabel, ylabel]

    plt.close('all')
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(3, 3))
    fig2, ax2 = plt.subplots(1, 1, squeeze=True)

    train_keep_all = df_this.index.get_level_values('train_keep').unique()
    # num_layer_all = merged_main.index.get_level_values('num_layer').unique()
    train_keep_max = train_keep_all.max()
    assert set(train_keep_all) <= {train_keep_max, train_keep_max // 2, train_keep_max // 4}

    limit_max, limit_min = df_this.quantile(0.975).max(), df_this.quantile(0.025).min()
    assert limit_max > limit_min
    limit_diff = limit_max - limit_min
    limit_max = limit_max + 0.1 * limit_diff
    limit_min = limit_min - 0.1 * limit_diff
    limit = (limit_min, limit_max)

    label_name = hyperparameter_name_friendly if hyperparameter_name_friendly is not None else hyperparameter_name

    for train_keep in train_keep_all:
        df_this_train_keep = df_this.xs(train_keep, level='train_keep')

        t_test = ttest_rel(
            df_this_train_keep[xlabel].values, df_this_train_keep[ylabel].values,
        )
        print(train_keep, t_test)

        ax.scatter(
            x=df_this_train_keep[xlabel].values,
            y=df_this_train_keep[ylabel].values,
            label=str(100 * train_keep // train_keep_max) + '%' + ', n={}'.format(
                df_this_train_keep.shape[0]) + ', p={:.2f}'.format(t_test.pvalue),
            s=6,
        )

        diff = df_this_train_keep[ylabel].values - df_this_train_keep[xlabel].values
        ax2.hist(
            diff,
            label=str(100 * train_keep // train_keep_max) + '%' + ', {:.2f}'.format(diff.mean()),
            bins=10,
            alpha=0.5
        )

    # ax.axis('equal')
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    ax.set_xlim(*limit)
    ax.set_ylim(*limit)
    ax.plot(limit, limit, linestyle='--', color='k')
    # ax.set_xlabel(f'{label_name}={xlabel}')
    # ax.set_ylabel(f'{label_name}={ylabel}')
    if hy_value_mapping is not None:
        xlabel = hy_value_mapping[xlabel]
        ylabel = hy_value_mapping[ylabel]

    fig.text(0.5, 0.0, f'{label_name}={xlabel}', ha='center', va='bottom')
    fig.text(0.0, 0.5, f'{label_name}={ylabel}', va='center', rotation='vertical', ha='left')

    ax2.axvline(0, linestyle='--')
    ax2.set_xlabel(f'{label_name}: {ylabel}-{xlabel}')
    ax.legend(
        loc='upper left',
        borderaxespad=0., fontsize='x-small', bbox_to_anchor=(0.01, 0.99), handletextpad=0
    )
    ax2.legend()

    if title is not None:
        ax.set_title(title)
        ax2.set_title(title)

    if plot_dir is not None:
        assert plot_name is not None
        makedirs(plot_dir, exist_ok=True)
        savefig(fig, key=join(plot_dir, plot_name + '.pdf'))

    plt.show()


def reduce_stuffs(df_to_process):
    axes_to_reduce = ['act_fn', 'ff_1st_bn_before_act', 'loss_type']
    key_and_possible_values = {
        k: df_to_process.index.get_level_values(k).unique().values.tolist() for k in axes_to_reduce
    }
    return process_r(
        df_to_process, key_and_possible_values, avg_inner, True
    ), reduce(
            lambda x, y: x* y,
            (len(x) for x in key_and_possible_values.values()),
            1
    )


def plot_additional_plots(
        *, df_in, plot_dir, metric_list, max_cls=None
):
    all_add_data = main_loop_for_additional_tables(
        df_in=df_in,
        max_cls=max_cls,
        metric_list=metric_list
    )

    hyperparameter_to_check = [
        'act_fn', 'loss_type', 'num_layer', 'ff_1st_bn_before_act'
    ]
    data_to_use = all_add_data['hyperameter_data']
    assert data_to_use.keys() == set(metric_list)
    for metric in metric_list:
        print(metric)
        data_to_use_this_metric = data_to_use[metric]

        # plot bars
        # n_per_bar is n models aggregated over seeds.
        merged_reduced, n_per_bar = reduce_stuffs(data_to_use_this_metric['total_merged'])

        for readout_mode in merged_reduced.index.get_level_values('readout_type').unique():
            merged_reduced_this = merged_reduced.xs(readout_mode, level='readout_type').copy(deep=True)

            train_keep_all = merged_reduced_this.index.get_level_values('train_keep').unique()
            # num_layer_all = merged_main.index.get_level_values('num_layer').unique()
            train_keep_max = train_keep_all.max()
            assert set(train_keep_all) <= {train_keep_max, train_keep_max // 2, train_keep_max // 4}

            unique_num_param = sorted(merged_reduced_this['num_param_mean_mean'].unique())
            # make sure that we can indeed sort models by model size
            assert (
                    len(unique_num_param) * merged_reduced_this.index.get_level_values(
                'train_keep'
            ).unique().size == merged_reduced_this.shape[0]
            )
            gen = ('{:02d},{}ch,{}Rl {}K'.format(unique_num_param.index(param_num), num_ch, (num_layer - 1) // 2,
                                                 int(round(param_num / 1000))) for num_ch, num_layer, param_num in
                   zip(merged_reduced_this.index.get_level_values('out_channel'),
                       merged_reduced_this.index.get_level_values('num_layer'),
                       merged_reduced_this['num_param_mean_mean']))
            merged_reduced_this['model_name'] = list(gen)

            merged_reduced_this = merged_reduced_this.reset_index(
                ['out_channel', 'num_layer'], drop=True
            ).set_index('model_name', append=True, verify_integrity=True)

            ccc = merged_reduced_this[['improvement_rel_mean', 'improvement_rel_sem']].unstack('model_name')
            plt.close('all')
            fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(4, 3))
            fig.subplots_adjust(left=0.2, right=0.99, bottom=0.2, top=0.9)
            series_mean = ccc['improvement_rel_mean']
            series_sem = ccc['improvement_rel_sem']
            series_mean.plot(ax=ax, kind='bar', yerr=series_sem)
            labels = ax.get_legend_handles_labels()[1]
            ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.99, 0.99),
                      borderaxespad=0., fontsize='x-small', handletextpad=0,
                      labels=[x[3:] for x in labels]
                      )
            assert series_mean.index.values.tolist() == series_sem.index.values.tolist() == sorted(train_keep_all)
            ax.set_title(readout_type_mapping[readout_mode] + f', n={n_per_bar} per bar')
            ax.set_xticklabels(
                [
                    str(100 * x // train_keep_max) + '%' for x in series_mean.index.values.tolist()
                ], rotation = 0
            )
            ax.set_ylabel('%$\\Delta$ of ' + metric_dict[metric])
            ax.set_xlabel('amount of training data')
            plt.show()
            if plot_dir is not None:
                savefig(fig, join(plot_dir, f'{metric}_by_size_{readout_mode}.pdf'))

        for hy in hyperparameter_to_check:
            hy_name_friendly = {
                'loss_type': 'loss fn',
                'act_fn': 'act layer',
                'num_layer': '# of layers',
                'ff_1st_bn_before_act': 'BN before act in 1st layer'
            }[hy]

            hy_value_mapping = {
                'num_layer': {
                    3: '1 R layer',
                    5: '2 R layers'
                }
            }.get(hy, None)
            # plot ff
            print('ff')
            compare_one_hyperpameter(
                df_merged=data_to_use_this_metric['df_ff'],
                hyperparameter_name=hy, column='perf_mean',
                plot_dir=plot_dir,
                hyperparameter_name_friendly=hy_name_friendly,
                title=metric_dict[metric] + ' of FF models',
                plot_name=f'{metric}_perf_mean_{hy}_ff',
            )

            compare_one_hyperpameter_over_readout_types(
                df_merged=data_to_use_this_metric['total_merged'],
                hyperparameter_name=hy, column='improvement_abs',
                metric=metric, hyperparameter_name_friendly=hy_name_friendly,
                plot_dir=plot_dir,
                hy_value_mapping=hy_value_mapping,
            )

            compare_one_hyperpameter_over_readout_types(
                df_merged=data_to_use_this_metric['total_merged'],
                hyperparameter_name=hy, column='improvement_rel',
                metric=metric, hyperparameter_name_friendly=hy_name_friendly,
                plot_dir=plot_dir,
                hy_value_mapping=hy_value_mapping,
            )
