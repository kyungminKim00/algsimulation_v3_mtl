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
import header.index_forecasting.RUNHEADER as RUNHEADER
import numpy as np
import pandas as pd
import ray
import statsmodels.api as sm
import tensorflow as tf
from header.index_forecasting import RUNHEADER
from sklearn.preprocessing import RobustScaler
from util import (
    _remove_cond,
    _replace_cond,
    current_y_unit,
    dict2json,
    find_date,
    funTime,
    ordinary_return,
)

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

warnings.filterwarnings("ignore")


# def cv_index_configuration(date, verbose):
#     num_per_shard = int(math.ceil(len(date) / float(_NUM_SHARDS)))
#     start_end_index_list = np.zeros([_NUM_SHARDS, 2])  # start and end index
#     if verbose == 0:  # train and validation
#         for shard_id in range(_NUM_SHARDS):
#             start_end_index_list[shard_id] = [
#                 shard_id * num_per_shard,
#                 min((shard_id + 1) * num_per_shard, len(date)),
#             ]
#     else:
#         start_end_index_list[0] = [0, len(date)]  # from 0 to end

#     return _cv_index_configuration(start_end_index_list, verbose), verbose


# def _cv_index_configuration(start_end_index_list, verbose):
#     index_container = list()
#     validation = list()
#     train = list()
#     if verbose == 0:  # train and validation
#         for idx in range(len(start_end_index_list)):
#             for ckeck_idx in range(len(start_end_index_list)):
#                 if ckeck_idx == idx:
#                     validation.append(start_end_index_list[ckeck_idx])
#                 else:
#                     train.append(start_end_index_list[ckeck_idx])
#             index_container.append([validation, train])
#             validation = list()
#             train = list()
#     else:
#         index_container = start_end_index_list
#     return index_container


def check_nan(data, keys):
    check = np.argwhere(np.sum(np.isnan(data), axis=0) == 1)
    if len(check) > 0:
        raise ValueError(
            "{0} contains nan values".format(keys[check.reshape(len(check))])
        )


def get_conjunction_dates_data(sd_dates, y_index_dates, sd_data, y_index_data):
    sd_dates_true = np.empty(0, dtype=np.int)
    y_index_dates_true = np.empty(0, dtype=np.int)
    y_index_dates_true_label = np.empty(0, dtype=np.object)

    for i in range(len(sd_dates)):
        for k in range(len(y_index_dates)):
            if (
                sd_dates[i] == y_index_dates[k]
            ):  # conjunction of sd_dates and y_index_dates
                if np.sum(np.isnan(y_index_data[:, 0])) == 0:
                    sd_dates_true = np.append(sd_dates_true, i)
                    y_index_dates_true = np.append(y_index_dates_true, k)
                    y_index_dates_true_label = np.append(
                        y_index_dates_true_label, y_index_dates[k]
                    )

    sd_dates = sd_dates[sd_dates_true]
    sd_data = sd_data[sd_dates_true]

    y_index_dates = y_index_dates[y_index_dates_true]

    assert len(sd_dates) == len(y_index_dates)
    assert len(sd_dates) == len(y_index_data)
    check_nan(sd_data, np.arange(sd_data.shape[1]))
    check_nan(y_index_data, np.arange(y_index_data.shape[1]))

    return sd_dates, sd_data, y_index_data


def get_conjunction_dates_data_v3(sd_dates, y_index_dates, sd_data, y_index_data):
    assert len(sd_dates) == len(sd_data), "length check"
    assert len(y_index_dates) == len(y_index_data), "length check"
    assert len(np.argwhere(np.isnan(sd_data))) == 0, ValueError("data contains nan")
    assert y_index_dates.ndim == sd_dates.ndim, "check dimension"
    assert y_index_dates.ndim == 1, "check dimension"

    def _get_conjunction_dates_data_v3(s_dates, t_dates, t_data):
        conjunctive_idx = [np.argwhere(t_dates == _dates) for _dates in s_dates]
        conjunctive_idx = sorted(
            [it[0][0] for it in conjunctive_idx if it.shape[0] == 1]
        )
        return t_data[conjunctive_idx], t_dates[conjunctive_idx]

    # y_index_data, ref = remove_nan(
    #     y_index_data, target_col=RUNHEADER.m_target_index, axis=0
    # )
    # if len(ref) > 0:
    #     y_index_dates = np.delete(y_index_dates, ref)

    sd_data, sd_dates = _get_conjunction_dates_data_v3(y_index_dates, sd_dates, sd_data)
    y_index_data, y_index_dates = _get_conjunction_dates_data_v3(
        sd_dates, y_index_dates, y_index_data
    )
    assert np.sum(sd_dates == y_index_dates) == len(y_index_dates), "check it"
    assert len(sd_data) == len(y_index_data), "check it"

    sd_data = np.array(sd_data, dtype=np.float32)
    y_index_data = np.array(y_index_data, dtype=np.float32)

    check_nan(sd_data, np.arange(sd_data.shape[1]))
    check_nan(y_index_data, np.arange(y_index_data.shape[1]))

    return sd_dates, sd_data, y_index_dates, y_index_data


# def load_file(file_location, file_format):
#     with open(file_location, "rb") as fp:
#         if file_format == "npy":
#             return np.load(fp)
#         elif file_format == "pkl":
#             return pickle.load(fp)
#         else:
#             raise ValueError("non-support file format")


def get_working_dates(dates, data):
    """Retrieve working days
    Args:
    path : raw data path

    """
    assert dates.shape[0] == data.shape[0], "the number of rows are different"

    # the data from monday to friday
    working_days_index = list()
    for i in range(len(dates)):
        tmp_date = datetime.datetime.strptime(dates[i], "%Y-%m-%d")
        if tmp_date.weekday() < 5:  # keep working days
            working_days_index.append(i)
        dates[i] = tmp_date.strftime("%Y-%m-%d")

    dates = dates[working_days_index]  # re-store working days
    data = data[working_days_index]  # re-store working days
    assert dates.shape[0] == data.shape[0], "the number of rows are different"

    return dates, data


# def replace_inf(values):
#     return _replace_cond(np.isinf, values)


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

    # # get working dates
    # index_dates, index_values = get_working_dates(index_dates, index_values)
    # y_index_dates, y_index_values = get_working_dates(y_index_dates, y_index_values)

    # replace nan with forward fill
    # index_values = replace_nan(index_values)

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


# def _pool_adhoc1(data, ids_to_var_names, opt="None", th=0.975):
#     return get_uniqueness(
#         from_file=False, _data=data, _dict=ids_to_var_names, opt=opt, th=th
#     )


# def _pool_adhoc2(data, ids_to_var_names):
#     return unit_datetype.quantising_vars(data, ids_to_var_names)


@ray.remote
def ray_wrap_fun(fun, ma_data, x_idx, num_cov_obs):
    return rolling_apply_cross_cov(fun, ma_data[:, x_idx], ma_data[:, -1], num_cov_obs)[
        :, 0, 1
    ]


@ray.remote
def ray_wrap_lr(X, Y):
    result = sm.OLS(Y, X).fit()
    return [np.abs(result.params[0]), result.rsquared]


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

        # the variable selection with cross corealation
        print(
            f"[{ma_data.shape[1] - 1} vars with {RUNHEADER.target_id2name(y_idx)}] cross corrleation"
        )
        res = [
            ray_wrap_fun.remote(fun_cross_cov, ma_data, x_idx, num_cov_obs)
            for x_idx in range(ma_data.shape[1] - 1)
        ]
        new_cov = np.array(ray.get(res)).T

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
        mean_coef, mean_rs = bn.nanmean(lr_res, axis=0)
        lr_dict = dict(zip(list(ids_to_var_names.values()), lr_res.tolist()))
        lr_dict = OrderedDict(
            [
                [key, np.abs(val[0])]
                for key, val in lr_dict.items()
                if (val[1] > mean_rs) and (val[0] > mean_coef)
            ]
        )
        lr_dict = OrderedDict(sorted(lr_dict.items(), key=lambda x: x[1], reverse=True))

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
        len(dates) - 70
    )  # for operation, it has been changed after a experimental
    RUNHEADER.m_pool_sample_end = len(dates)
    num_sample_obs = [RUNHEADER.m_pool_sample_start, RUNHEADER.m_pool_sample_end]
    num_cov_obs = 25  # default 20
    max_allowed_num_variables = 25  # default 20 각 시장별 20개 변수 선택 15*20 -> 300
    explane_th = RUNHEADER.explane_th
    plot = True  # default False
    opts = None
    var_names_to_ids = dict(
        zip(list(ids_to_var_names.values()), list(ids_to_var_names.keys()))
    )

    def _save(_dates, _data, _ids_to_var_names):
        file_name = RUNHEADER.file_data_vars + "total_market"
        _data = np.append(np.expand_dims(_dates, axis=1), _data, axis=1)
        # print("{} saving".format(file_name))

        _mode = ".csv"
        pd.DataFrame(data=list(_ids_to_var_names.values()), columns=["VarName"]).to_csv(
            file_name + "_Indices.csv", index=None, header=None
        )

        # rewrite
        unit_datetype.script_run(file_name + "_Indices.csv")

        # pd.DataFrame(
        #     data=_data, columns=["TradeDate"] + list(_ids_to_var_names.values())
        # ).to_csv(file_name + _mode, index=None)
        # print("save done {} ".format(file_name + _mode))
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
