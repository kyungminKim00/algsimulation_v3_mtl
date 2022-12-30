from __future__ import absolute_import, division, print_function

import datetime
import math
import os
import pickle
import sys
import time
import warnings
from collections import OrderedDict

import bottleneck as bn
import numpy as np
import pandas as pd
import ray
import statsmodels.api as sm
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

import datasets.unit_datetype_des_check as unit_datetype
from datasets import dataset_utils
from datasets.dataset_utils import bytes_feature, float_feature, int64_feature
from datasets.decoder import pkexample_type_A
from datasets.windowing import (
    fun_cov,
    fun_cross_cov,
    fun_cumsum,
    fun_mean,
    rolling_apply,
    rolling_apply_cov,
    rolling_apply_cross_cov,
)
from datasets.x_selection import get_uniqueness, get_uniqueness_without_dates
from header.index_forecasting import RUNHEADER
from util import (
    _remove_cond,
    _replace_cond,
    current_y_unit,
    dict2json,
    find_date,
    funTime,
    get_conjunction_dates_data_v3,
    get_working_dates,
    ordinary_return,
)


@ray.remote
def ray_wrap_fun(fun, ma_data, x_idx, num_cov_obs):
    return rolling_apply_cross_cov(fun, ma_data[:, x_idx], ma_data[:, -1], num_cov_obs)[
        :, 0, 1
    ]


@ray.remote
def ray_wrap_lr(X, Y):
    warnings.filterwarnings("ignore")
    result = sm.OLS(Y, X).fit()
    return [np.abs(result.params[0]), result.rsquared]


def data_from_csv(filename, eod=None):
    index_df = pd.read_csv(filename)
    assert (
        index_df["TradeDate"].isnull().sum() == 0
    ), "check the column of dataframe, dataset shoud be invlove the 'TradeDate' column and does not contain nan values "
    index_df = index_df.ffill(axis=0)
    index_df = index_df.bfill(axis=0)
    if eod is not None:
        _dates = index_df.values
        e_test_idx = (
            find_date(_dates, eod, -1)
            if len(np.argwhere(_dates == eod)) == 0
            else np.argwhere(_dates == eod)[0][0]
        )
        index_df = index_df.iloc[e_test_idx - 750 : e_test_idx, :]
    else:
        index_df = index_df.iloc[-750:, :]

    dates, data = get_working_dates(
        index_df.values[:, 0], np.array(index_df.values[:, 1:], dtype=np.float32)
    )
    ids_to_class_names = OrderedDict(
        zip(range(len(index_df.keys()[1:])), index_df.keys()[1:])
    )
    return dates, data, ids_to_class_names


def get_data_corresponding(index_price, y_index, eod=None):
    index_dates, index_values, ids_to_var_names = data_from_csv(index_price, eod)
    y_index_dates, y_index_values, ids_to_class_names = data_from_csv(y_index, eod)

    # align dates of target and independent variables (the conjunction of target and independent variables)
    dates, data, y_index_dates, y_index_data = get_conjunction_dates_data_v3(
        index_dates, y_index_dates, index_values, y_index_values
    )

    return (
        dates,
        data,
        y_index_dates,
        y_index_data,
        ids_to_var_names,
        ids_to_class_names,
    )


def splite_rawdata_v1(index_price=None, y_index=None, eod=None):
    (dates, sd_data, _, y_index_data, ids_to_var_names, _,) = get_data_corresponding(
        index_price,
        y_index,
        eod=eod,
    )

    # up to 0.97 S&P500, NASDAQ, DOW
    # up to 0.75 KOSPI, KOSDAQ
    print(
        f"Apply uniqueness with 0.94 to var:{sd_data.shape[1]} smaples:{sd_data.shape[0]}"
    )
    sd_data, ids_to_var_names = get_uniqueness_without_dates(
        from_file=False, _data=sd_data, _dict=ids_to_var_names, opt="mva", th=0.94
    )
    print(f"result shape: {sd_data.shape}")

    gen_pool(
        dates,
        sd_data,
        ids_to_var_names,
        y_index_data,
    )


def pool_ordering_refine(
    data,
    target_data,
    ids_to_var_names,
    var_names_to_ids,
    base_first_momentum,
    num_sample_obs,
    num_cov_obs,
    max_allowed_num_variables,
    explane_th,
):
    num_y_var = target_data.shape[1]
    ordered_ids_list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for y_idx in range(num_y_var):
            data = np.hstack([data, np.expand_dims(target_data[:, y_idx], axis=1)])

            # data = np.concatenate([data, target_data[:, idx]], axis=1)

            # latest_3y_samples = num_sample_obs[1] - (20 * 12 * 3)

            ma_data = bn.move_mean(
                data[num_sample_obs[0] : num_sample_obs[1], :],
                window=base_first_momentum,
                min_count=1,
                axis=0,
            )

            # the variable selection with cross corealation - part 1 pivot to calculate mean_cor_pivot, std_cor_pivot
            print(
                f"[{ma_data.shape[1] - 1} vars with {RUNHEADER.target_id2name(y_idx)}] cross corrleation"
            )
            res = [
                ray_wrap_fun.remote(fun_cross_cov, ma_data, x_idx, num_cov_obs)
                for x_idx in range(ma_data.shape[1] - 1)
            ]
            new_cov = np.array(ray.get(res)).T
            mean_cor_pivot, std_cor_pivot = np.nanmean(new_cov, axis=0), np.nanstd(
                new_cov, axis=0
            )

            # determin marginal area
            upper = mean_cor_pivot + std_cor_pivot * RUNHEADER.var_select_factor
            lower = mean_cor_pivot - std_cor_pivot * RUNHEADER.var_select_factor

            # # use a half of samples
            # # the variable selection with cross corealation - part 2 to calculate mean_cor
            new_cov = new_cov[-RUNHEADER.m_pool_samples :, :]
            mean_cor = np.nanmean(new_cov, axis=0)

            # liner regression
            tmp_cov = np.where(np.isnan(new_cov), 0, new_cov)
            res = [
                ray_wrap_lr.remote(
                    np.arange(tmp_cov.shape[0]),
                    tmp_cov[:, var_idx],
                )
                for var_idx in range(tmp_cov.shape[1])
            ]
            lr_res = np.array(ray.get(res))
            mean_coef, mean_rs = np.nanmean(lr_res, axis=0)  # basket coef and rs

            lr_res = np.concatenate(
                [
                    lr_res,
                    mean_cor.reshape([-1, 1]),
                    upper.reshape([-1, 1]),
                    lower.reshape([-1, 1]),
                ],
                axis=1,
            )

            lr_dict = dict(zip(list(ids_to_var_names.values()), lr_res.tolist()))
            lr_dict = OrderedDict(
                [
                    [key, np.abs(val[0])]
                    for key, val in lr_dict.items()
                    # if (val[0] > mean_coef) and ((val[2] > val[3]) or (val[2] < val[4]))
                    if ((val[1] < mean_rs) and (val[0] > mean_coef))
                    and ((val[2] > val[3]) or (val[2] < val[4]))
                ]
            )
            lr_dict = OrderedDict(
                sorted(lr_dict.items(), key=lambda x: x[1], reverse=True)
            )

            # 2-3. Re-assign Dict & Data
            ordered_ids = [var_names_to_ids[name] for name in lr_dict.keys()]
            # 2-3-1. Apply max_num of variables
            print(f"the num of variables exceeding explane_th: {len(ordered_ids)}")
            num_variables = len(ordered_ids)
            if num_variables > max_allowed_num_variables:
                ordered_ids = ordered_ids[:max_allowed_num_variables]
            print(
                f"the num of selected variables {len(ordered_ids)} from {num_variables} for {RUNHEADER.target_id2name(y_idx)}"
            )
            ordered_ids_list = ordered_ids_list + ordered_ids

            # for Monitoring Service
            save_name = f"{RUNHEADER.file_data_vars}{RUNHEADER.target_id2name(y_idx)}"
            pd.DataFrame(
                data=[ids_to_var_names[ids] for ids in ordered_ids], columns=["VarName"]
            ).to_csv(save_name + "_Indices.csv", index=None, header=None)
            # rewrite
            unit_datetype.script_run(save_name + "_Indices.csv")

    ordered_ids = list(set(ordered_ids_list))
    # 2-3-2. re-assign
    ids_to_var_names = OrderedDict(
        zip(
            np.arange(len(ordered_ids)).tolist(),
            [ids_to_var_names[ids] for ids in ordered_ids],
        )
    )
    var_names_to_ids = dict(
        zip(list(ids_to_var_names.values()), list(ids_to_var_names.keys()))
    )

    data = data[:, :-1]
    data = data.T[ordered_ids].T

    return data, ids_to_var_names, var_names_to_ids


def gen_pool(dates, sd_data, ids_to_var_names, target_data):
    base_first_momentum = 5  # default 5
    # RUNHEADER.m_pool_sample_start = (len(dates) - 750)  # for operation, it has been changed after a experimental
    RUNHEADER.m_pool_sample_start = (
        len(dates) - RUNHEADER.m_pool_samples * 2
    )  # for operation, it has been changed after a experimental  # 140
    RUNHEADER.m_pool_sample_end = len(dates)
    num_sample_obs = [RUNHEADER.m_pool_sample_start, RUNHEADER.m_pool_sample_end]
    num_cov_obs = 20  # default 20
    max_allowed_num_variables = 25  # default 25 각 시장별 20개 변수 선택 15*25
    explane_th = RUNHEADER.explane_th
    plot = True  # default False
    opts = None
    var_names_to_ids = dict(
        zip(list(ids_to_var_names.values()), list(ids_to_var_names.keys()))
    )

    def _save(_dates, _data, _ids_to_var_names):
        file_name = RUNHEADER.file_data_vars + RUNHEADER.target_id2name(
            RUNHEADER.m_target_index
        )

        # file_name = RUNHEADER.file_data_vars + "TOTAL"
        _data = np.append(np.expand_dims(_dates, axis=1), _data, axis=1)

        pd.DataFrame(data=list(_ids_to_var_names.values()), columns=["VarName"]).to_csv(
            file_name + "_Indices.csv", index=None, header=None
        )

        # rewrite
        unit_datetype.script_run(file_name + "_Indices.csv")

        os._exit(0)

    data, ids_to_var_names, var_names_to_ids = pool_ordering_refine(
        sd_data,
        target_data,
        ids_to_var_names,
        var_names_to_ids,
        base_first_momentum,
        num_sample_obs,
        num_cov_obs,
        max_allowed_num_variables,
        explane_th,
    )

    assert len(dates) == data.shape[0], "Type Check!!!"
    assert len(ids_to_var_names) == data.shape[1], "Type Check!!!"

    print("Pool Refine Done!!!")
    _save(dates, data, ids_to_var_names)


def run(dataset_dir, file_pattern="fs_v0_cv%02d_%s.tfrecord", s_test=None, e_test=None):

    index_price: str = RUNHEADER.raw_x
    y_index: str = RUNHEADER.raw_y

    splite_rawdata_v1(index_price=index_price, y_index=y_index, eod=e_test)
