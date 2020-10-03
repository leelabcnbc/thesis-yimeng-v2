from collections import defaultdict

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt


def get_mean(x):
    if x is None:
        return np.nan
    return x['diffs'].mean()


def get_sem(x):
    if x is None:
        return np.nan
    return sem(x['diffs'], ddof=0)

def print_one(*, data_baseline, data_learned, title=None):
    if title is not None:
        print(title)
    assert data_baseline.shape == data_learned.shape
    assert data_baseline.ndim == data_learned.ndim == 1
    print('baseline mean {:.4f}, sem {:.4f}'.format(data_baseline.mean(), sem(data_baseline, ddof=0)))
    print('learned mean {:.4f}, sem {:.4f}'.format(data_learned.mean(), sem(data_learned, ddof=0)))
    print('% of pairs, learned > mean {:.2%}'.format(((data_learned-data_baseline)>0).mean()))


def final_one_case(
        *,
        data_baseline,
        data_learned,
        save_name=None):
    # plot a final one
    plt.close('all')

    data_baseline_all = []
    data_learned_all = []

    assert data_baseline.keys() == data_learned.keys()
    for z1, z2 in zip(data_baseline.values(), data_learned.values()):
        data_baseline_all.extend(z1)
        data_learned_all.extend(z2)
    assert len(data_baseline) == len(data_learned)

    fig, axes = plt.subplots(nrows=len(data_learned)+1, ncols=2, figsize=(20, 4*(len(data_learned)+1)), squeeze=False)
    data_all = np.asarray([data_baseline_all, data_learned_all]).T
    print(data_all.shape)
    axes[0, 0].hist(data_all, label=['baseline', 'learned'], bins=20)
    axes[0, 0].legend()
    axes[0, 0].set_title('overall')

    axes[0, 1].scatter(data_baseline_all, data_learned_all, s=2)
    axes[0, 1].set_xlabel('baseline')
    axes[0, 1].set_ylabel('learned')
    axes[0, 1].plot([-0.2, 0.2], [-0.2, 0.2], linestyle='--', color='r')
    axes[0, 1].axis('equal')

    print_one(data_baseline=np.array(data_baseline_all), data_learned=np.asarray(data_learned_all), title='overall')

    # for following rows
    for axes_this, (x1, y1), (x2, y2) in zip(axes[1:], data_baseline.items(), data_learned.items()):
        assert x1 == x2
        axes_this[0].set_title(x1)
        axes_this[0].hist(np.asarray([y1, y2]).T, label=['baseline', 'learned'], bins=20)
        axes_this[0].legend()

        axes_this[1].scatter(y1, y2, s=2)
        axes_this[1].set_xlabel('baseline')
        axes_this[1].set_ylabel('learned')
        axes_this[1].plot([-0.2, 0.2], [-0.2, 0.2], linestyle='--', color='r')
        axes_this[1].axis('equal')

        print_one(data_baseline=np.array(y1), data_learned=np.asarray(y2), title=x1)

    plt.show()


def add_tuning_mean(df_main_result):
    df_main_result['hal_tuning_improved_mean'] = df_main_result['hal_tuning_analysis_improved'].map(
        lambda x: get_mean(x))
    df_main_result['hal_tuning_improved_baseline_mean'] = df_main_result['hal_tuning_analysis_improved_baseline'].map(
        lambda x: get_mean(x))
    df_main_result['hal_tuning_half_improved_mean'] = df_main_result['hal_tuning_analysis_half_improved'].map(
        lambda x: get_mean(x))
    df_main_result['hal_tuning_half_improved_baseline_mean'] = df_main_result[
        'hal_tuning_analysis_half_improved_baseline'].map(lambda x: get_mean(x))


def show_scatter_plots(df_main_result):
    add_tuning_mean(df_main_result)
    diff_mean_half_improved_all = defaultdict(list)
    diff_mean_half_improved_all_baseline = defaultdict(list)
    diff_mean_improved_all = defaultdict(list)
    diff_mean_improved_all_baseline = defaultdict(list)
    for train_keep in df_main_result.index.get_level_values('train_keep').unique():
        df_main_result_this_train = df_main_result.xs(train_keep, level='train_keep')
        for cls in df_main_result_this_train.index.get_level_values('rcnn_bl_cls').unique():
            if cls == 1:
                continue

            df_this_main = df_main_result_this_train.xs(cls, level='rcnn_bl_cls')

            for readout_type in df_this_main.index.get_level_values('readout_type').unique():
                df_this_readout = df_this_main.xs(readout_type, level='readout_type')
                #                 plt.close('all')
                #                 fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                #                 axes = axes.ravel()

                #                 axes[0].scatter(
                #                     df_this_readout['hal_tuning_improved_baseline_mean'].values,
                #                     df_this_readout['hal_tuning_improved_mean'].values,
                #                     alpha=0.5,
                #                     s=8
                #                 )
                #                 axes[0].plot([-0.2, 0.2], [-0.2, 0.2], linestyle='--')
                #                 axes[1].scatter(
                #                     df_this_readout['hal_tuning_half_improved_baseline_mean'].values,
                #                     df_this_readout['hal_tuning_half_improved_mean'].values,
                #                     alpha=0.5,
                #                     s=8
                #                 )
                #                 axes[1].plot([-0.2, 0.2], [-0.2, 0.2], linestyle='--')

                diff_mean_improved_all[readout_type].extend(
                    df_this_readout['hal_tuning_improved_mean'].values.tolist()
                )
                diff_mean_improved_all_baseline[readout_type].extend(
                    df_this_readout['hal_tuning_improved_baseline_mean'].values.tolist(),
                )
                diff_mean_half_improved_all[readout_type].extend(
                    df_this_readout['hal_tuning_half_improved_mean'].values.tolist()
                )
                diff_mean_half_improved_all_baseline[readout_type].extend(
                    df_this_readout['hal_tuning_half_improved_baseline_mean'].values.tolist(),
                )

    final_one_case(data_baseline=diff_mean_improved_all_baseline, data_learned=diff_mean_improved_all)
    final_one_case(data_baseline=diff_mean_half_improved_all_baseline, data_learned=diff_mean_half_improved_all)
