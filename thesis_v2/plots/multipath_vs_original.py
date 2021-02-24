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
):
    depth_multi = get_depth(df_main_result).to_frame(name='depth_multi')
    depth_main = get_depth(df_main_result_ref).to_frame(name='depth_main')
    aaaa = depth_multi.join(depth_main, how='inner', lsuffix='', rsuffix='').sort_index()

    limit_max, limit_min = max(
        aaaa['depth_multi'].quantile(0.975), aaaa['depth_main'].quantile(0.975)
    ), min(
        aaaa['depth_multi'].quantile(0.025), aaaa['depth_main'].quantile(0.025)
    )
    assert limit_max > limit_min
    limit_diff = limit_max - limit_min
    limit_max = limit_max + 0.1 * limit_diff
    limit_min = limit_min - 0.1 * limit_diff
    limit = (limit_min, limit_max)

    for train_keep in aaaa.index.get_level_values('train_keep').unique():
        plt.close('all')
        fig, ax = plt.subplots(squeeze=True, figsize=(4, 4))
        b = aaaa.xs(train_keep, level='train_keep')
        n = b.shape[0]
        r = pearsonr(b['depth_main'].values, b['depth_multi'].values)[0]
        ax.scatter(b['depth_main'].values, b['depth_multi'].values, s=1,
                   label='training data {}, n={}, r={:.2f}'.format(
                       training_data_mapping[train_keep], n, r
                   )
                   )

        # compute pearson
        print(train_keep)

        ax.set_xlabel('''average path length of original recurrent models''')
        ax.set_ylabel('''average path length multi-path models''')
        ax.plot(limit, limit, linestyle='--', color='k')
        ax.set_xlim(*limit)
        ax.set_ylim(*limit)
        ax.legend()

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
):
    aaaa = df_main_result_ref.join(df_main_result.dropna(), how='inner', lsuffix='_ref', rsuffix='_new').sort_index()
    # check performance diff between two readout types
    plt.close('all')
    fig, ax = plt.subplots(squeeze=True, figsize=(4, 4))

    for train_keep in aaaa.index.get_level_values('train_keep').unique():
        b = aaaa.xs(train_keep, level='train_keep')
        n = b.shape[0]
        r = pearsonr(b['cc2_normed_avg_ref'].values, b['cc2_normed_avg_new'].values)[0]
        ax.scatter(b['cc2_normed_avg_ref'].values, b['cc2_normed_avg_new'].values, s=1,
                   label='training data {}, n={}, r={:.2f}'.format(
                       training_data_mapping[train_keep], n, r
                   )
                   )

        # compute pearson
        print(train_keep)

    ax.set_xlabel('''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$ of original recurrent models''')
    ax.set_ylabel('''average ${\\mathrm{CC}}_{\\mathrm{norm}}^2$ of multi-path models''')
    ax.plot([0, 1], [0, 1], linestyle='--', color='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    makedirs(pltdir, exist_ok=True)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    savefig(fig, join(pltdir, 'multi_vs_r.pdf'))

    plt.show()
