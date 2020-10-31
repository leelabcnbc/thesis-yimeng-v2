"""
a rewrite of `main_results.py`, in more idiomatic Pandas, and directly output what I really want
"""
from typing import List
from copy import deepcopy
from itertools import product
from functools import partial
import pandas as pd
import numpy as np


def avg_inner(df, *, key_and_possible_values):
    # act_fn, ff_1st_bn_before_act, loss_type, model_seed
    # should cover everything.
    keys = sorted(key_and_possible_values)
    values = zip(*[df.index.get_level_values(k) for k in keys])
    values_ref = product(*[key_and_possible_values[k] for k in keys])
    assert set(values) == set(values_ref)

    # then compute mean
    # then compute sem.
    return pd.concat([df.mean(axis=0).rename(lambda x: x + '_mean'),
                      df.sem(axis=0, ddof=0).rename(lambda x: x + '_sem')])


def reduce_df(df_this, key_and_possible_values):
    return df_this.groupby(level=[x for x in df_this.index.names if x not in key_and_possible_values]).apply(
        partial(avg_inner, key_and_possible_values=key_and_possible_values)
    )


def process_ff(df_ff_this: pd.DataFrame, key_and_possible_values):
    # average out all different GPU cards (same parameter, different runs using different cards)
    readout_type_parsed: pd.DataFrame = df_ff_this.groupby(
        level=[x for x in df_ff_this.index.names if x != 'readout_type']).mean()
    # then average out seeds
    return reduce_df(readout_type_parsed, key_and_possible_values)


def process_r(df_r_this, key_and_possible_values):
    return reduce_df(df_r_this, key_and_possible_values)


def merge_thin_and_wide(*, df_fewer_columns, df_more_columns, fewer_suffix):
    # DO NOT use JOIN, or index.
    # reset_index and then set back!!!!! pandas.DataFrame.join's logic is pretty complicated.
    # df_r's columns
    # out_channel, rcnn_bl_cls, readout_type, train_keep, num_layer_ff
    # df_ff's columns
    # num_layer_ff, out_channel, train_keep
    #
    #
    assert df_more_columns.index.is_unique
    assert df_fewer_columns.index.is_unique

    r_index_names = df_more_columns.index.names
    ff_index_names = df_fewer_columns.index.names
    assert set(r_index_names) >= set(ff_index_names)
    df_fewer_columns = df_fewer_columns.reset_index()
    df_more_columns = df_more_columns.reset_index()
    merged = df_more_columns.merge(df_fewer_columns, on=ff_index_names, how='inner', suffixes=('', fewer_suffix))
    return merged.set_index(r_index_names, verify_integrity=True).sort_index()


def preprocess(df_in, *, max_cls, axes_to_reduce, override_ff_num_layer=None):
    # provide main table
    columns = df_in.columns.tolist()
    df_in = df_in.copy()
    if max_cls is not None:
        df_in = df_in[df_in.index.get_level_values('rcnn_bl_cls') <= max_cls]

    # separate into ff and non-ff
    assert (df_in.index.get_level_values('rcnn_bl_cls') >= 1).all()
    df_ff = df_in.xs(key=1, level='rcnn_bl_cls')
    df_r = df_in[df_in.index.get_level_values('rcnn_bl_cls') != 1]

    # map num_layer in df_r to the corresponding FF layers.
    df_r = df_r.reset_index('num_layer')
    df_r['num_layer'] = (df_r['num_layer'] - 1) * 2 + 1
    df_r = df_r.set_index('num_layer', append=True)

    # then check the possible values taken over axes_to_reduce
    # I use r's values.
    key_and_possible_values = {
        k: df_r.index.get_level_values(k).unique().values.tolist() for k in axes_to_reduce
    }

    key_and_possible_values_ff = deepcopy(key_and_possible_values)
    if 'num_layer' in key_and_possible_values_ff and override_ff_num_layer is not None:
        key_and_possible_values_ff['num_layer'] = override_ff_num_layer

    # filter ff table
    for kk, vv in key_and_possible_values_ff.items():
        df_ff = df_ff[df_ff.index.get_level_values(kk).isin(vv)]
    df_ff = process_ff(df_ff, key_and_possible_values_ff)
    df_r = process_r(df_r, key_and_possible_values)
    return columns, df_ff, df_r


def get_perf_over_cls_data(df_in: pd.DataFrame, *,
                           axes_to_reduce: List[str], display, max_cls: int = None):
    columns, df_ff, df_r = preprocess(df_in, max_cls=max_cls, axes_to_reduce=axes_to_reduce)
    total_merged = merge_thin_and_wide(df_more_columns=df_r, df_fewer_columns=df_ff, fewer_suffix='_ff')

    # create ff vs perf table, one for each column c
    #
    # ecah column c has two tables
    # ff vs different cls for c_mean
    # ff vs different cls for c_sem
    good_order = [
        'train_keep',
        'out_channel',
        'num_layer',
        'readout_type'
    ]
    for c in columns:
        for subtype in ['mean', 'sem']:
            print(f'ff vs different cls, {c}_{subtype}')
            col_ff = f'{c}_{subtype}_ff'
            col_r = f'{c}_{subtype}'
            ff_part = total_merged[col_ff].xs(key=2, level='rcnn_bl_cls').to_frame(name='ff')
            # compute max gain
            max_gain = ((total_merged[col_r] - total_merged[col_ff]) / total_merged[col_ff]).unstack(
                'rcnn_bl_cls'
            )
            assert np.all(np.isfinite(max_gain.values))
            max_gain = max_gain.max(axis=1) * 100
            r_part = total_merged[col_r].unstack('rcnn_bl_cls')
            assert np.all(np.isfinite(r_part.values))
            final_this = pd.concat([ff_part,
                                    r_part,
                                    max_gain.to_frame('max gain %')
                                    ], axis=1)

            # remove the name of columns
            # https://stackoverflow.com/questions/29765548/remove-index-name-in-pandas
            final_this = final_this.rename_axis(None, axis=1)
            # order columns
            final_this = final_this.reorder_levels(
                order=[x for x in good_order if x in final_this.index.names],
                axis=0
            ).sort_index()
            display(final_this)


def get_ff_vs_best_r_data(
        df_in: pd.DataFrame, *,
        axes_to_reduce: List[str], display, max_cls: int = None,
        reference_num_layer_ff: int,
        override_ff_num_layer: List[int]
):
    # provide ff table
    assert 'num_layer' not in axes_to_reduce
    assert reference_num_layer_ff in override_ff_num_layer
    columns, df_ff, df_r = preprocess(df_in, max_cls=max_cls, axes_to_reduce=axes_to_reduce,
                                      override_ff_num_layer=override_ff_num_layer)
    assert reference_num_layer_ff % 2 == 1
    # `num_layer` in df_r has been transformed
    df_r = df_r.xs(key=reference_num_layer_ff, level='num_layer')
    df_fewer = df_r.unstack(['rcnn_bl_cls', 'readout_type'])
    assert np.all(np.isfinite(df_fewer.values))
    total_merged = merge_thin_and_wide(
        df_fewer_columns=df_fewer.max(axis=1, level=0),
        df_more_columns=df_ff, fewer_suffix='_r')

    good_order = [
        'train_keep',
        'out_channel',
        'num_layer',
        'readout_type'
    ]
    for c in columns:
        for subtype in ['mean', 'sem']:
            print(f'best vs different num_layer, {c}_{subtype}')
            col_ff = f'{c}_{subtype}'
            col_r = f'{c}_{subtype}_r'
            ff_part = total_merged[col_ff].unstack('num_layer')
            assert np.all(np.isfinite(ff_part.values))
            r_part = total_merged[col_r].xs(key=reference_num_layer_ff, level='num_layer').to_frame(name='max r')

            # compute max gain
            max_gain_r = (
                    ((r_part['max r'] - ff_part[reference_num_layer_ff]) / ff_part[reference_num_layer_ff]) * 100
            ).to_frame(name='max r gain %')

            # compute max gain ff
            max_gain_ff = []
            for zzz in ff_part.columns:
                if zzz == reference_num_layer_ff:
                    continue
                pp = (ff_part[zzz] - ff_part[reference_num_layer_ff]) / ff_part[reference_num_layer_ff] * 100
                max_gain_ff.append(pp.to_frame(name=zzz))
            max_gain_ff = pd.concat(max_gain_ff, axis=1).max(axis=1).to_frame(name='max FF gain %')

            final_this = pd.concat([ff_part,
                                    max_gain_ff,
                                    max_gain_r
                                    ], axis=1)
            # remove the name of columns
            # https://stackoverflow.com/questions/29765548/remove-index-name-in-pandas
            final_this = final_this.rename_axis(None, axis=1)
            # order columns
            final_this = final_this.reorder_levels(
                order=[x for x in good_order if x in final_this.index.names],
                axis=0
            ).sort_index()
            display(final_this)
