# code for ablation study
# adapted from code in /results_processed/tang_refactored/maskcnn_polished_with_rcnn_k_bl/20201218_plot.ipynb

from collections.abc import Iterable
from itertools import chain

import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.stats import entropy


def get_perf(df, perf_col):
    df = remove_high_cls(df).sort_index()
    assert df.index.unique
    return df[perf_col].dropna().sort_index()


def get_depth(df):
    df = remove_high_cls(df).dropna().sort_index()
    assert df.index.unique
    df['source_analysis_vec'] = df['source_analysis'].map(lambda x: get_normalized_vec(x))
    df['source_analysis_vec_scalar'] = df['source_analysis_vec'].map(lambda x: get_weighted_avg(x))
    return df['source_analysis_vec_scalar']


def get_depth_entropy(df):
    df = remove_high_cls(df).dropna().sort_index()
    assert df.index.unique
    df['source_analysis_vec'] = df['source_analysis'].map(lambda x: get_normalized_vec(x))
    df['source_analysis_vec_entropy'] = df['source_analysis_vec'].map(lambda x: entropy(x))
    return df['source_analysis_vec_entropy']


def get_depth_distribution(df):
    df = remove_high_cls(df).dropna().sort_index()
    assert df.index.unique
    df['source_analysis_vec'] = df['source_analysis'].map(lambda x: get_normalized_vec(x))
    return df['source_analysis_vec']


def remove_high_cls(df_this):
    df_this = df_this[df_this.index.get_level_values('rcnn_bl_cls') <= 7]
    return df_this.sort_index()


def get_normalized_vec(x):
    #     keys = [('I',) + ('B1',) + ('R1',)*i for i in range(7)]
    #     assert x.keys() <= set(keys)
    ret = np.zeros((8,))
    for key, v in x.items():
        assert len(key) >= 2 and len(key) <= 9
        ret[len(key) - 2] += v  # -2 because minimal length of key is 2 (I + one Conv)
    ret = ret / ret.sum()
    return ret


# source_analysis_vec_scalar is sum(i*w for i, w in enumerage(vec))
# average depth.

def get_weighted_avg(x):
    assert x.ndim == 1 and x.shape == (8,) and np.all(x >= 0)
    return np.average(np.arange(8), weights=x)


def get_common_index(*, df_perf, df_depth, train_keep, num_layer):
    # avoid unnecessary join. note that here `num_layer` is the original one,
    # so a 2 layer recurrent model has 2 here, not 3.
    df_perf = df_perf.xs((train_keep, num_layer), level=('train_keep', 'num_layer')).sort_index()
    df_depth = df_depth.xs((train_keep, num_layer), level=('train_keep', 'num_layer')).sort_index()
    assert df_perf.index.names == df_depth.index.names
    assert df_perf.index.equals(df_depth.index)
    return df_perf.index


def get_geD1_perf(perf):
    ret = perf.xs('geD1', level='multi_path_hack')
    return ret.sort_index().rename('geD1')


def get_leDX_perf(perf):
    ret = perf[
        perf.index.get_level_values(
            'multi_path_hack'
            # here .values is important. otherwise this code does not work properly. not sure why.
            # pandas is complicated.
        ).map(lambda x: x.startswith('leD')).values & (
                perf.index.get_level_values('multi_path_hack').map(lambda x: int(x[3:])) == perf.index.get_level_values(
            'rcnn_bl_cls')
        )
        ]

    return ret.droplevel('multi_path_hack').sort_index().rename('leDX')


def get_baseline_perf(*, perf, perf_baseline):
    # this is average of geD1, leDX, as well as original models trained in 20201114+20201118.
    df1 = get_geD1_perf(perf)
    df2 = get_leDX_perf(perf)
    df3 = perf_baseline
    assert df1.index.names == df2.index.names
    assert df1.index.names == df3.index.names
    # take average
    merged = pd.concat([df1.sort_index(), df2.sort_index(), df3.sort_index()], axis=1)
    return merged.mean(axis=1).dropna()


def check_index(data_source_data, *, trim_data=None):
    assert data_source_data.keys() == {'perf', 'depth'}
    assert data_source_data['perf'].index.equals(
        data_source_data['depth'].index
    )
    assert data_source_data['perf'].index.names == data_source_data['depth'].index.names
    if trim_data is not None:
        for level, value in trim_data.items():
            if not isinstance(value, Iterable):
                # https://stackoverflow.com/a/1952655
                data_source_data['perf'] = data_source_data['perf'].xs(value, level=level)
                data_source_data['depth'] = data_source_data['depth'].xs(value, level=level)
            else:
                data_source_data['perf'] = data_source_data['perf'][
                    data_source_data['perf'].index.get_level_values(level).isin(value)
                ]
                data_source_data['depth'] = data_source_data['depth'][
                    data_source_data['depth'].index.get_level_values(level).isin(value)
                ]
        data_source_data['perf'] = data_source_data['perf'].sort_index()
        data_source_data['depth'] = data_source_data['depth'].sort_index()
        check_index(data_source_data)


def collect_all_data(
        *,
        data_loader_dict,
        train_keep,
        num_layer=2,
        perf_col='cc2_normed_avg',
        out_channel=(16, 32),
):
    assert num_layer == 2
    # only models studied here will be further studied.
    reference_source = 'multipath'
    # load all data.
    # handle each readout_type
    data_all = dict()
    for data_source, data_loader in data_loader_dict.items():
        data_all[data_source] = {
            'perf': get_perf(data_loader['perf'](), perf_col),
            'depth': get_depth(data_loader['depth']())
        }
        check_index(data_all[data_source],
                    trim_data={'train_keep': train_keep, 'num_layer': num_layer,
                               'out_channel': out_channel}
                    )

    # for multipath baseline, hack data to get correct baseline
    # this is needed because some models in multipath baseline is not avaliable due to some transient error.
    # also, averaging over multiple run should be good.
    data_all[reference_source]['perf'] = get_baseline_perf(
        perf=data_all['leDXgeDX']['perf'],
        perf_baseline=data_all['multipath']['perf'],
    )

    data_all[reference_source]['depth'] = get_baseline_perf(
        perf=data_all['leDXgeDX']['depth'],
        perf_baseline=data_all['multipath']['depth'],
    )
    check_index(data_all[reference_source])
    reference_index = data_all[reference_source]['perf'].index.sort_values()
    print(reference_index.shape)

    # collect data for every type.
    result_all = []
    data_source_all = []
    for data_source_this, data_source_data in data_all.items():
        result_all.append(
            collect_all_data_inner(
                data_source=data_source_this,
                data=data_source_data,
                reference_index=reference_index,
            )
        )
        data_source_all.append(data_source_this)

    return pd.concat(result_all, axis=0,
                     keys=data_source_all,
                     verify_integrity=True,
                     names=['data_source']).sort_index()


def collect_all_data_inner(
        *,
        data_source,
        data,
        reference_index,
):
    # over each readout type
    result_all = []

    df_combined = pd.concat(list(v.to_frame(k) for k, v in data.items()), axis=1)
    assert df_combined.columns.tolist() == ['perf', 'depth']

    aggregate_level_all = []

    for readout_type in chain(reference_index.get_level_values('readout_type').unique(), [None]):
        # get data in this slice.
        if readout_type is not None:
            df_combined_this = df_combined.xs(readout_type, level='readout_type')
            _, reference_index_this = reference_index.get_loc_level(readout_type, level='readout_type')
            reference_index_this = reference_index_this.sort_values()
        else:
            df_combined_this = df_combined
            reference_index_this = reference_index

        result_all.append(
            collect_all_data_inner_helper(
                data_source=data_source,
                df_combined=df_combined_this,
                reference_index=reference_index_this,
            )
        )
        aggregate_level_all.append(
            f'readout_type={readout_type}' if readout_type is not None else ''
        )

    return pd.concat(result_all, axis=0,
                     keys=aggregate_level_all,
                     verify_integrity=True,
                     names=['aggregate_level'])


def collect_all_data_inner_helper(
        *,
        data_source,
        df_combined,
        reference_index,
):
    result_all = []
    for rcnn_bl_cls in reference_index.get_level_values('rcnn_bl_cls').unique():
        result_all.append(
            collect_all_data_inner_per_cls(
                data_source=data_source,
                df_combined=df_combined.xs(rcnn_bl_cls, level='rcnn_bl_cls'),
                reference_index=reference_index.get_loc_level(rcnn_bl_cls, level='rcnn_bl_cls')[1].sort_values(),
                rcnn_bl_cls=rcnn_bl_cls,
            )
        )
        result_all[-1]['rcnn_bl_cls'] = rcnn_bl_cls

    ret = pd.DataFrame(
        result_all, columns=sorted(result_all[0].keys()),
    ).set_index(['rcnn_bl_cls'], verify_integrity=True)
    return ret


def collect_all_data_inner_per_cls(
        *,
        data_source,
        df_combined,
        reference_index,
        # this is needed for leDXgeDX, onlyDX
        rcnn_bl_cls,
):
    data = []
    if data_source == 'onlyDX':
        for d in range(1, rcnn_bl_cls + 1):
            data.append(
                collect_one_cls(
                    df_combined=df_combined.xs(
                        f'onlyD{d}', level='multi_path_hack'
                    ),
                    reference_index=reference_index,
                )
            )
    elif data_source == 'leDXgeDX':
        for d_lower in range(1, rcnn_bl_cls):
            data.append(
                collect_one_cls(
                    df_combined=df_combined.xs(
                        f'leD{d_lower}', level='multi_path_hack'
                    ),
                    reference_index=reference_index,
                )
            )
        for d_higher in range(2, rcnn_bl_cls + 1):
            data.append(
                collect_one_cls(
                    df_combined=df_combined.xs(
                        f'geD{d_higher}', level='multi_path_hack'
                    ),
                    reference_index=reference_index,
                )
            )
    else:
        # simple. just collect.
        data.append(collect_one_cls(
            df_combined=df_combined,
            reference_index=reference_index,
        ))

    ret = {
        'perf_mean': [],
        'perf_sem': [],
        'depth_mean': [],
        'depth_sem': [],
        'n': None,
    }
    for v in data:
        assert ret.keys() == v.keys()
        if ret['n'] is None:
            ret['n'] = v['n']
        assert ret['n'] == v['n']
        for k1, v1 in v.items():
            if k1 == 'n':
                continue
            ret[k1].append(v1)
    return ret


def collect_one_cls(
        *,
        df_combined, reference_index,
):
    assert df_combined.index.names == reference_index.names
    assert reference_index.isin(df_combined.index).all()
    data = df_combined.loc[reference_index]

    ret = dict()
    ret['n'] = reference_index.size
    for col in data:
        values = data[col].values
        assert values.ndim == 1
        assert np.all(np.isfinite(values))
        ret[col + '_mean'] = values.mean()
        ret[col + '_sem'] = sem(values, ddof=0)

    return ret
