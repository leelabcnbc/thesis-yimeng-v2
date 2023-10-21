from os import makedirs
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .util import savefig
from ..analysis.ablation_study import get_depth


def plot_scatter_multi_path_characteristics(
        *,
        df_main_result,
        df_main_result_ref,
        pltdir,
        training_data_mapping,
        nips_hacking=False,
):
    depth_multi = get_depth(df_main_result).to_frame(name='depth_multi')
    depth_main = get_depth(df_main_result_ref).to_frame(name='depth_main')
    aaaa = depth_multi.join(depth_main, how='inner', lsuffix='', rsuffix='').sort_index()

    limit_max, limit_min = max(
        aaaa['depth_multi'].quantile(0.975), aaaa['depth_main'].quantile(0.975)
    ), min(
        aaaa['depth_multi'].quantile(0.025), aaaa['depth_main'].quantile(0.025)
    )

    # https://stackoverflow.com/a/42091037
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    assert limit_max > limit_min
    limit_diff = limit_max - limit_min
    limit_max = limit_max + 0.1 * limit_diff
    limit_min = limit_min - 0.1 * limit_diff
    limit = (limit_min, limit_max)

    for idx, train_keep in enumerate(aaaa.index.get_level_values('train_keep').unique()):
        print(idx)
        plt.close('all')
        fig, ax = plt.subplots(squeeze=True, figsize=(4, 4))
        b = aaaa.xs(train_keep, level='train_keep')
        n = b.shape[0]
        r = pearsonr(b['depth_main'].values, b['depth_multi'].values)[0]
        ax.scatter(b['depth_main'].values, b['depth_multi'].values, s=1,
                   label='training data {}, n={}, r={:.2f}'.format(
                       training_data_mapping[train_keep], n, r,
                   ),
                   c=colors[idx]
                   )

        # compute pearson
        print(train_keep)

        fontdict = {
            'fontsize': 'x-large'
        } if nips_hacking else None

        xlabel = {
            True: 'avg path lengh, R',
            False: '''average path length of original recurrent models''',
        }[nips_hacking]

        ylabel = {
            True: 'avg path lengh, multi-path',
            False: '''average path length multi-path models''',
        }[nips_hacking]

        ax.set_xlabel(xlabel, fontdict=fontdict)
        ax.set_ylabel(ylabel, fontdict=fontdict)
        ax.plot(limit, limit, linestyle='--', color='k')
        ax.set_xlim(*limit)
        ax.set_ylim(*limit)

        if not nips_hacking:
            ax.legend()
        else:
            ax.text(
                0, 1, s='n={}, r={:.2f}'.format(n, r), horizontalalignment='left',
                verticalalignment='top', fontsize='xx-large',
                transform=ax.transAxes,
            )
            ax.set_xticks([1,2,3,4,5])
            ax.set_yticks([1,2,3,4,5])
            ax.set_xticklabels(['1', '2', '3', '4', '5'], fontdict={'fontsize': 'x-large'})
            ax.set_yticklabels(['1', '2', '3', '4', '5'], fontdict={'fontsize': 'x-large'})


        makedirs(pltdir, exist_ok=True)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
        savefig(fig, join(pltdir, f'multi_vs_r_depth_{train_keep}.pdf'))

        plt.show()


def plot_scatter(
        *,
        df_main_result,
        df_main_result_ref,
        pltdir,
        training_data_mapping,
        nips_hacking=False,
):
    aaaa = df_main_result_ref.join(df_main_result.dropna(), how='inner', lsuffix='_ref', rsuffix='_new').sort_index()
    # check performance diff between two readout types
    plt.close('all')
    fig, ax = plt.subplots(squeeze=True, figsize=(4, 4))

    for train_keep in aaaa.index.get_level_values('train_keep').unique():

        # if nips_hacking:
        #     # designed for NS 2250 data set.
        #     if train_keep != 1400:
        #         continue

        b = aaaa.xs(train_keep, level='train_keep')
        n = b.shape[0]
        r = pearsonr(b['cc2_normed_avg_ref'].values, b['cc2_normed_avg_new'].values)[0]
        ax.scatter(b['cc2_normed_avg_ref'].values, b['cc2_normed_avg_new'].values, s=1,
                   label='training data {}, n={}, r={:.2f}'.format(
                       training_data_mapping[train_keep], n, r
                   ),
                   )

        # compute pearson
        print(train_keep)


    fontdict = {
        'fontsize': 'x-large'
    } if nips_hacking else None

    xlabel = {
        False: '''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$ of original recurrent models''',
        True: '''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$, R''',
    }[nips_hacking]

    ylabel = {
        False: '''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$ of multi-path models''',
        True: '''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$,\nmulti-path''',
    }[nips_hacking]

    ax.set_xlabel(xlabel, fontdict=fontdict)
    ax.set_ylabel(ylabel, fontdict=fontdict, labelpad=(-12.5 if nips_hacking else None))
    ax.plot([0, 1], [0, 1], linestyle='--', color='k')

    if not nips_hacking:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.set_xlim(0.35, 0.55)
        ax.set_ylim(0.35, 0.55)
        ax.set_xticks([0.4, 0.5])
        ax.set_yticks([0.4, 0.5])
        ax.set_xticklabels(['0.4', '0.5'], fontdict={'fontsize': 'x-large'})
        ax.set_yticklabels(['0.4', '0.5'], rotation=90, fontdict={'fontsize': 'x-large'})
        ax.text(
            0, 1, s='n={}, r={:.2f}'.format(n, r), horizontalalignment='left',
            verticalalignment='top', fontsize='xx-large',
            transform=ax.transAxes,
        )

    if not nips_hacking:
        ax.legend()

    makedirs(pltdir, exist_ok=True)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    savefig(fig, join(pltdir, 'multi_vs_r.pdf'))

    plt.show()
