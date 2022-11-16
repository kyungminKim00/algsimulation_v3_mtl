"""Converts data to TFRecords of TF-Example protos.

This module creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers

The script should take about a minute to run.

"""

from __future__ import absolute_import, division, print_function

import datetime
import math
import os
import pickle
import sys
from collections import OrderedDict

import bottleneck as bn
import header.index_forecasting.RUNHEADER as RUNHEADER
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from util import (
    _remove_cond,
    _replace_cond,
    current_y_unit,
    get_working_dates,
    ordinary_return,
    trans_val,
)

from datasets import dataset_utils
from datasets.convert_if_v1_common import (
    ReadData,
    add_data_4_operation,
    cut_off_data,
    cv_index_configuration,
    get_corr,
    load_file,
    ma,
    normalized_spread,
    splite_rawdata_v1,
)
from datasets.decoder import pkexample_type_A, pkexample_type_B
from datasets.unit_datetype_des_check import write_var_desc

# import tf_slim as slim
# slim.variable


# def cv_index_configuration(date, verbose):
#     num_per_shard = int(math.ceil(len(date) / float(_NUM_SHARDS)))
#     start_end_index_list = np.zeros([_NUM_SHARDS, 2])  # start and end index
#     if verbose == 0:  # train and validation separately
#         for shard_id in range(_NUM_SHARDS):
#             start_end_index_list[shard_id] = [
#                 shard_id * num_per_shard,
#                 min((shard_id + 1) * num_per_shard, len(date)),
#             ]
#     elif verbose == 1:  # test
#         start_end_index_list[0] = [0, len(date)]
#     elif verbose == 2:  # from 0 to end - only train without validation
#         start_end_index_list[0] = [0, len(date)]
#     elif verbose == 3:  # duplicated validation for early stopping criteria
#         headbias_from_y_excluded = forward_ndx + ref_forward_ndx[-1]
#         duplicated_samples = (
#             -40 - headbias_from_y_excluded - RUNHEADER.m_warm_up_4_inference
#         )
#         start_end_index_list[0] = [0, len(date)]
#         start_end_index_list[1] = [len(date) + duplicated_samples, len(date)]
#     elif verbose == 4:  # train and validation separately
#         headbias_from_y_excluded = forward_ndx + ref_forward_ndx[-1]
#         val_samples = -250 - headbias_from_y_excluded  # val samples 1years
#         start_end_index_list[0] = [0, len(date) - val_samples]
#         start_end_index_list[1] = [len(date) + val_samples, len(date)]
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
#     elif verbose == 1:
#         index_container = start_end_index_list
#     elif verbose == 2:
#         index_container = start_end_index_list
#     elif (
#         verbose == 3 or verbose == 4
#     ):  # index_container contains validation and training
#         index_container.append([start_end_index_list[1], start_end_index_list[0]])
#     return index_container


def convert_dataset(
    sd_dates,
    sd_data,
    sd_ma_data_5,
    sd_ma_data_10,
    sd_ma_data_20,
    sd_ma_data_60,
    sd_diff_data,
    sd_diff_ma_data_5,
    sd_diff_ma_data_10,
    sd_diff_ma_data_20,
    sd_diff_ma_data_60,
    sd_velocity_data,
    sd_velocity_ma_data_5,
    sd_velocity_ma_data_10,
    sd_velocity_ma_data_20,
    sd_velocity_ma_data_60,
    historical_ar_data,
    historical_ar_ma_data_5,
    historical_ar_ma_data_10,
    historical_ar_ma_data_20,
    historical_ar_ma_data_60,
    target_data,
    fund_his_data_30,
    fund_cov_data_60,
    extra_cov_data_60,
    mask,
    x_seq,
    class_names_to_ids,
    dataset_dir,
    verbose,
    forward_ndx=None,
    ref_forward_ndx=None,
    decoder=None,
):
    """Converts the given filenames to a TFRecord - tf.train.examples."""

    date = sd_dates

    # Data Binding.. initialize data helper class
    sd_reader = ReadData(
        date,
        sd_data,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_reader_ma5 = ReadData(
        date,
        sd_ma_data_5,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_reader_ma10 = ReadData(
        date,
        sd_ma_data_10,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_reader_ma20 = ReadData(
        date,
        sd_ma_data_20,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_reader_ma60 = ReadData(
        date,
        sd_ma_data_60,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_reader = ReadData(
        date,
        sd_diff_data,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_reader_ma5 = ReadData(
        date,
        sd_diff_ma_data_5,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_reader_ma10 = ReadData(
        date,
        sd_diff_ma_data_10,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_reader_ma20 = ReadData(
        date,
        sd_diff_ma_data_20,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_reader_ma60 = ReadData(
        date,
        sd_diff_ma_data_60,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_reader = ReadData(
        date,
        sd_velocity_data,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_reader_ma5 = ReadData(
        date,
        sd_velocity_ma_data_5,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_reader_ma10 = ReadData(
        date,
        sd_velocity_ma_data_10,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_reader_ma20 = ReadData(
        date,
        sd_velocity_ma_data_20,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_reader_ma60 = ReadData(
        date,
        sd_velocity_ma_data_60,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )
    mask_reader = ReadData(
        date,
        mask,
        target_data,
        x_seq,
        class_names_to_ids,
        forward_ndx,
        ref_forward_ndx,
    )

    # Data set configuration - generate cross validation index
    index_container, verbose = cv_index_configuration(
        date, verbose, forward_ndx, ref_forward_ndx
    )

    return _convert_dataset(
        date,
        sd_reader,
        sd_reader_ma5,
        sd_reader_ma10,
        sd_reader_ma20,
        sd_reader_ma60,
        sd_diff_reader,
        sd_diff_reader_ma5,
        sd_diff_reader_ma10,
        sd_diff_reader_ma20,
        sd_diff_reader_ma60,
        sd_velocity_reader,
        sd_velocity_reader_ma5,
        sd_velocity_reader_ma10,
        sd_velocity_reader_ma20,
        sd_velocity_reader_ma60,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        mask_reader,
        x_seq,
        index_container,
        dataset_dir,
        verbose,
        forward_ndx=forward_ndx,
        ref_forward_ndx=ref_forward_ndx,
        decoder=decoder,
    )


def _convert_dataset(
    date,
    sd_reader,
    sd_reader_ma5,
    sd_reader_ma10,
    sd_reader_ma20,
    sd_reader_ma60,
    sd_diff_reader,
    sd_diff_reader_ma5,
    sd_diff_reader_ma10,
    sd_diff_reader_ma20,
    sd_diff_reader_ma60,
    sd_velocity_reader,
    sd_velocity_reader_ma5,
    sd_velocity_reader_ma10,
    sd_velocity_reader_ma20,
    sd_velocity_reader_ma60,
    historical_ar_reader,
    historical_ar_reader_ma5,
    historical_ar_reader_ma10,
    historical_ar_reader_ma20,
    historical_ar_reader_ma60,
    fund_his_reader_30,
    fund_cov_reader_60,
    extra_cov_reader_60,
    mask_reader,
    x_seq,
    index_container,
    dataset_dir,
    verbose,
    forward_ndx=None,
    ref_forward_ndx=None,
    decoder=None,
):
    with tf.Graph().as_default():
        if verbose == 1:  # verbose=1 for test
            test_list = index_container[[0]]
            return write_patch(
                sd_reader,
                sd_reader_ma5,
                sd_reader_ma10,
                sd_reader_ma20,
                sd_reader_ma60,
                sd_diff_reader,
                sd_diff_reader_ma5,
                sd_diff_reader_ma10,
                sd_diff_reader_ma20,
                sd_diff_reader_ma60,
                sd_velocity_reader,
                sd_velocity_reader_ma5,
                sd_velocity_reader_ma10,
                sd_velocity_reader_ma20,
                sd_velocity_reader_ma60,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                mask_reader,
                x_seq,
                test_list,
                "test",
                1,
                train_sample=False,
                forward_ndx=forward_ndx,
                ref_forward_ndx=ref_forward_ndx,
                decoder=decoder,
            )
        if verbose == 3 or verbose == 4:
            validation_list = [index_container[0][0]]

            # for validation
            return write_patch(
                sd_reader,
                sd_reader_ma5,
                sd_reader_ma10,
                sd_reader_ma20,
                sd_reader_ma60,
                sd_diff_reader,
                sd_diff_reader_ma5,
                sd_diff_reader_ma10,
                sd_diff_reader_ma20,
                sd_diff_reader_ma60,
                sd_velocity_reader,
                sd_velocity_reader_ma5,
                sd_velocity_reader_ma10,
                sd_velocity_reader_ma20,
                sd_velocity_reader_ma60,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                mask_reader,
                x_seq,
                validation_list,
                "validation",
                1,
                train_sample=False,
                forward_ndx=forward_ndx,
                ref_forward_ndx=ref_forward_ndx,
                decoder=decoder,
            )
        assert False, "None defiend verbose"


def write_patch(
    sd_reader,
    sd_reader_ma5,
    sd_reader_ma10,
    sd_reader_ma20,
    sd_reader_ma60,
    sd_diff_reader,
    sd_diff_reader_ma5,
    sd_diff_reader_ma10,
    sd_diff_reader_ma20,
    sd_diff_reader_ma60,
    sd_velocity_reader,
    sd_velocity_reader_ma5,
    sd_velocity_reader_ma10,
    sd_velocity_reader_ma20,
    sd_velocity_reader_ma60,
    historical_ar_reader,
    historical_ar_reader_ma5,
    historical_ar_reader_ma10,
    historical_ar_reader_ma20,
    historical_ar_reader_ma60,
    fund_his_reader_30,
    fund_cov_reader_60,
    extra_cov_reader_60,
    mask_reader,
    x_seq,
    index_container,
    output_filename,
    stride,
    train_sample=True,
    forward_ndx=None,
    ref_forward_ndx=None,
    decoder=None,
):
    # Get patch
    pk_data = list()
    data_set_mode = [
        dn for dn in ["test", "train", "validation"] if dn in output_filename
    ][0]

    for _, val in enumerate(index_container):  # iteration with contained span lists
        start_ndx, end_ndx = val
        start_ndx, end_ndx = int(start_ndx), int(end_ndx)  # type casting
        for i in range(start_ndx, end_ndx, stride):
            if train_sample:
                sample_criteria_dummy_1 = x_seq * 2
                sample_criteria_dummy = forward_ndx + ref_forward_ndx[-1]
            else:
                sample_criteria_dummy_1 = (
                    x_seq if x_seq > ref_forward_ndx[-1] else ref_forward_ndx[-1]
                )
                sample_criteria_dummy_1 = (
                    forward_ndx
                    if forward_ndx > sample_criteria_dummy_1
                    else sample_criteria_dummy_1
                )
                sample_criteria_dummy_1 = (
                    sample_criteria_dummy_1 + 5
                )  # for his return of x
                sample_criteria_dummy = forward_ndx
            # Read Data
            if ((i - sample_criteria_dummy_1) >= 0) and (
                (i + sample_criteria_dummy) < end_ndx
            ):
                sd_reader.get_patch(i, train_sample)
                sd_reader_ma5.get_patch(i, train_sample)
                sd_reader_ma10.get_patch(i, train_sample)
                sd_reader_ma20.get_patch(i, train_sample)
                sd_reader_ma60.get_patch(i, train_sample)
                sd_diff_reader.get_patch(i, train_sample)
                sd_diff_reader_ma5.get_patch(i, train_sample)
                sd_diff_reader_ma10.get_patch(i, train_sample)
                sd_diff_reader_ma20.get_patch(i, train_sample)
                sd_diff_reader_ma60.get_patch(i, train_sample)
                sd_velocity_reader.get_patch(i, train_sample)
                sd_velocity_reader_ma5.get_patch(i, train_sample)
                sd_velocity_reader_ma10.get_patch(i, train_sample)
                sd_velocity_reader_ma20.get_patch(i, train_sample)
                sd_velocity_reader_ma60.get_patch(i, train_sample)
                mask_reader.get_patch(i, train_sample)

                # when only support pickle, e.g. mpi
                pk_data.append(
                    decoder(
                        sd_reader,
                        sd_reader_ma5,
                        sd_reader_ma10,
                        sd_reader_ma20,
                        sd_reader_ma60,
                        sd_diff_reader,
                        sd_diff_reader_ma5,
                        sd_diff_reader_ma10,
                        sd_diff_reader_ma20,
                        sd_diff_reader_ma60,
                        sd_velocity_reader,
                        sd_velocity_reader_ma5,
                        sd_velocity_reader_ma10,
                        sd_velocity_reader_ma20,
                        sd_velocity_reader_ma60,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        mask_reader,
                        data_set_mode,
                        RUNHEADER.pkexample_type["num_features_1"],
                        RUNHEADER.pkexample_type["num_features_2"],
                        RUNHEADER.pkexample_type["num_market"],
                    )
                )
    return pk_data


# def check_nan(data, keys):
#     check = np.argwhere(np.sum(np.isnan(data), axis=0) == 1)
#     if len(check) > 0:
#         raise ValueError(
#             f"{keys[check.reshape(len(check))] contains nan values")


# def get_conjunction_dates_data(sd_dates, y_index_dates, sd_data, y_index_data):
#     sd_dates_true = np.empty(0, dtype=np.int)
#     y_index_dates_true = np.empty(0, dtype=np.int)
#     y_index_dates_true_label = np.empty(0, dtype=np.object)

#     for i in range(len(sd_dates)):
#         for k in range(len(y_index_dates)):
#             if (
#                 sd_dates[i] == y_index_dates[k]
#             ):  # conjunction of sd_dates and y_index_dates
#                 if np.sum(np.isnan(y_index_data[:, 0])) == 0:
#                     sd_dates_true = np.append(sd_dates_true, i)
#                     y_index_dates_true = np.append(y_index_dates_true, k)
#                     y_index_dates_true_label = np.append(
#                         y_index_dates_true_label, y_index_dates[k]
#                     )

#     sd_dates = sd_dates[sd_dates_true]
#     sd_data = sd_data[sd_dates_true]

#     y_index_dates = y_index_dates[y_index_dates_true]

#     assert len(sd_dates) == len(y_index_dates)
#     assert len(sd_dates) == len(y_index_data)
#     check_nan(sd_data, np.arange(sd_data.shape[1]))
#     check_nan(y_index_data, np.arange(y_index_data.shape[1]))

#     return sd_dates, sd_data, y_index_data


# def get_conjunction_dates_data_v3(sd_dates, y_index_dates, sd_data, y_index_data):
#     assert len(sd_dates) == len(sd_data), "length check"
#     assert len(y_index_dates) == len(y_index_data), "length check"
#     assert len(np.argwhere(np.isnan(sd_data))) == 0, ValueError("data contains nan")
#     assert y_index_dates.ndim == sd_dates.ndim, "check dimension"
#     assert y_index_dates.ndim == 1, "check dimension"

#     def _get_conjunction_dates_data_v3(s_dates, t_dates, t_data):
#         conjunctive_idx = [np.argwhere(t_dates == _dates) for _dates in s_dates]
#         conjunctive_idx = sorted(
#             [it[0][0] for it in conjunctive_idx if it.shape[0] == 1]
#         )
#         return t_data[conjunctive_idx], t_dates[conjunctive_idx]

#     sd_data, sd_dates = _get_conjunction_dates_data_v3(y_index_dates, sd_dates, sd_data)
#     y_index_data, y_index_dates = _get_conjunction_dates_data_v3(
#         sd_dates, y_index_dates, y_index_data
#     )
#     assert np.sum(sd_dates == y_index_dates) == len(y_index_dates), "check it"
#     assert len(sd_data) == len(y_index_data), "check it"

#     sd_data = np.array(sd_data, dtype=np.float32)
#     y_index_data = np.array(y_index_data, dtype=np.float32)

#     check_nan(sd_data, np.arange(sd_data.shape[1]))
#     check_nan(y_index_data, np.arange(y_index_data.shape[1]))

#     return sd_dates, sd_data, y_index_dates, y_index_data


# def get_read_data(sd_dates, y_index_dates, sd_data, y_index_data):
#     """Validate data and Return actual operation days for target_index"""

#     # 1. [row-wised filter] the conjunction of structure data dates and fund-structure data dates
#     dates, sd_data, y_index_data = get_conjunction_dates_data(
#         sd_dates, y_index_dates, sd_data, y_index_data
#     )

#     return dates, sd_data, y_index_data


# def add_data_4_operation(data, test_e_date=None):
#     n_length = test_e_date - data.shape[0]

#     if data.ndim == 1:
#         add_data = np.zeros([n_length])
#     elif data.ndim == 2:
#         add_data = np.zeros([n_length, data.shape[1]])
#     elif data.ndim == 3:
#         add_data = np.zeros([n_length, data.shape[1], data.shape[2]])
#     else:
#         assert False, "check dimensions"
#     return np.append(data, add_data, axis=0)


# def load_file(file_location, file_format):
#     with open(file_location, "rb") as fp:
#         if file_format == "npy":
#             data = np.load(fp)
#             fp.close()
#             return data
#         elif file_format == "pkl":
#             data = pickle.load(fp)
#             fp.close()
#             return data
#         else:
#             raise ValueError("non-support file format")


# def replace_nan(values):
#     return _replace_cond(np.isnan, values)


# def replace_inf(values):
#     return _replace_cond(np.isinf, values)


# def remove_nan(values, target_col=None, axis=0):
#     return _remove_cond(np.isnan, values, target_col=target_col, axis=axis)


# def _get_index_df(v, index_price, ids_to_var_names, target_data=None):
#     x1, x2 = None, None
#     is_exist = False
#     for idx in range(len(ids_to_var_names)):
#         if "-" in v:
#             _v = v.split("-")
#             if _v[0] == ids_to_var_names[idx]:
#                 x1 = index_price[:, idx]
#             if _v[1] == ids_to_var_names[idx]:
#                 x2 = index_price[:, idx]
#             if (x1 is not None) and (x2 is not None):
#                 scale_v = np.append(
#                     np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1), axis=1
#                 )
#                 scale_v = np.hstack([scale_v, np.expand_dims(target_data, axis=1)])
#                 scale_v = RobustScaler().fit_transform(scale_v)
#                 # return np.abs(scale_v[:, 0] - scale_v[:, 1])
#                 return scale_v[:, 0] - scale_v[:, 1]
#         else:
#             if v == ids_to_var_names[idx]:
#                 return index_price[:, idx]

#     if not is_exist:
#         # assert is_exist, "could not find a given variable name: {}".format(v)
#         return np.zeros(index_price.shape[0])


# def get_index_df(
#     index_price=None, ids_to_var_names=None, c_name=None, target_data=None
# ):
#     index_df = [
#         _get_index_df(v, index_price, ids_to_var_names, target_data)
#         for v in c_name.values()
#     ]
#     index_df = np.array(index_df, dtype=np.float32).T

#     return np.array(index_df, dtype=np.float32), c_name


# def splite_rawdata_v1(index_price=None, y_index=None, c_name=None):
#     index_df = pd.read_csv(index_price)
#     index_dates = index_df.values[:, 0]
#     index_values = np.array(index_df.values[:, 1:], dtype=np.float32)
#     ids_to_var_names = OrderedDict(
#         zip(range(len(index_df.keys()[1:])), index_df.keys()[1:])
#     )

#     y_index_df = pd.read_csv(y_index)
#     y_index_dates = y_index_df.values[:, 0]
#     y_index_values = np.array(y_index_df.values[:, 1:], dtype=np.float32)
#     ids_to_class_names = OrderedDict(
#         zip(range(len(y_index_df.keys()[1:])), y_index_df.keys()[1:])
#     )

#     # get working dates
#     index_dates, index_values = get_working_dates(index_dates, index_values)
#     y_index_dates, y_index_values = get_working_dates(y_index_dates, y_index_values)

#     # the conjunction of target and independent variables
#     dates, sd_data, y_index_dates, y_index_data = get_conjunction_dates_data_v3(
#         index_dates, y_index_dates, index_values, y_index_values
#     )

#     unit = current_y_unit(RUNHEADER.target_name)
#     returns = ordinary_return(matrix=y_index_data, unit=unit)  # daily return

#     sd_data, ids_to_var_names = get_index_df(
#         sd_data, ids_to_var_names, c_name, y_index_data[:, RUNHEADER.m_target_index]
#     )

#     return dates, sd_data, y_index_data, returns, ids_to_class_names, ids_to_var_names


# def triangular_vector(data):
#     row, n_var, _ = data.shape
#     data = data.reshape(row, n_var**2)

#     # extract upper-triangular components
#     del_idx = list()
#     for n_idx in np.arange(n_var):
#         if n_idx == 0:
#             del_idx.append(0)
#         else:
#             for n_idx2 in np.arange(n_idx + 1):
#                 del_idx.append(n_idx * n_var + n_idx2)
#     triangular_idx = np.delete(np.arange(n_var**2), del_idx)

#     return data[:, triangular_idx]


# # dont move to util
# def _getcorr(
#     data, target_data, base_first_momentum, num_cov_obs, b_scaler=True, opt_mask=None
# ):
#     _data = np.hstack([data, np.expand_dims(target_data, axis=1)])
#     ma_data = bn.move_mean(
#         _data, window=base_first_momentum, min_count=1, axis=0
#     )  # use whole train samples

#     cov = rolling_apply_cov(fun_cov, ma_data, num_cov_obs, b_scaler)
#     cov = cov[:, :, -1]
#     cov = cov[:, :-1]

#     tmp_cov = np.where(np.isnan(cov), 0, cov)
#     tmp_cov = np.abs(tmp_cov)

#     daily_cov_raw = tmp_cov
#     tmp_cov = np.where(tmp_cov >= opt_mask, 1, 0)

#     return tmp_cov, daily_cov_raw


# # dont move to util
# def get_corr(data, target_data, x_unit=None, b_scaler=True, opt_mask=None):
#     base_first_momentum, num_cov_obs = 5, 40  # default

#     # 15번 y 흘려 주기
#     tmp_cov, daily_cov_raw = _getcorr(
#         data, target_data, base_first_momentum, num_cov_obs, b_scaler, opt_mask
#     )

#     if x_unit is not None:
#         add_vol_index = np.array(x_unit) == "volatility"
#         tmp_cov = add_vol_index + tmp_cov
#         tmp_cov = np.where(tmp_cov >= 1, 1, 0)

#     # mean_cov = np.nanmean(tmp_cov, axis=0)
#     # cov_dict = dict(zip(list(ids_to_var_names.values()), mean_cov.tolist()))
#     # cov_dict = OrderedDict(sorted(cov_dict.items(), key=lambda x: x[1], reverse=True))
#     total_num = int(tmp_cov.shape[1] * np.mean(np.mean(tmp_cov)))
#     print("the average num of variables on daily: {}".format(total_num))
#     mask = tmp_cov
#     return mask, daily_cov_raw


def configure_inference_dates(dates, s_test=None, e_test=None, forward_ndx=None):

    if len(dates) < e_test:  # OperationMode section
        s_test = len(dates) - 1
        dummy_dates_4_inference = list()
        datetime_obj = datetime.datetime.strptime(dates[s_test], "%Y-%m-%d")
        datetime_obj += datetime.timedelta(days=1)
        while True:
            if len(dummy_dates_4_inference) <= forward_ndx:
                if datetime_obj.weekday() < 5:
                    dummy_dates_4_inference.append(datetime_obj.strftime("%Y-%m-%d"))
                datetime_obj += datetime.timedelta(days=1)
            else:
                break
        dates_new = np.array(
            dates.tolist() + dummy_dates_4_inference[:-1], dtype=object
        )
        e_test = len(dates_new)
        operation_mode = True
    else:
        dates_new = dates
        operation_mode = False
    return dates_new, s_test, e_test, None, operation_mode


def run(
    x_dict,
    s_test=None,
    e_test=None,
    split_name=None,
    domain=None,
    _forward_ndx=None,
    opt_mask=None,
):
    """Conversion operation.
    Args:
    dataset_dir: The dataset directory where the dataset is stored.
    """

    index_price = RUNHEADER.raw_x  # S&P by jh
    y_index = RUNHEADER.raw_y
    # operation_mode = bool(operation_mode)

    ref_forward_ndx = np.array([-10, -5, 5, 10], dtype=np.int)
    ref_forward_ndx = np.array(
        [-int(_forward_ndx * 0.5), -int(_forward_ndx * 0.25), 5, 10], dtype=np.int
    )
    # performed_date = _performed_date

    """declare dataset meta information (part1)
    """
    x_seq = 20  # 20days
    forward_ndx = _forward_ndx
    cut_off = 70
    # num_of_datatype_obs = 5
    # num_of_datatype_obs_total = RUNHEADER.pkexample_type["num_features_1"]  # 25 -> 15
    # num_of_datatype_obs_total_mt = RUNHEADER.pkexample_type["num_features_2"]
    # RUNHEADER.m_warm_up_4_inference = int(forward_ndx)
    # RUNHEADER.m_warm_up_4_inference = 6

    dependent_var = "tri"
    # global g_num_of_datatype_obs

    decoder = globals()[RUNHEADER.pkexample_type["decoder"]]

    # if RUNHEADER.use_var_mask:
    #     decoder = pkexample_type_B
    # else:
    #     decoder = pkexample_type_A

    # # var_names for the target instrument
    # if RUNHEADER.use_c_name:
    #     # manually selected with analysis
    #     if RUNHEADER.re_assign_vars:
    #         c_name = "{}{}_Indices.csv".format(
    #             RUNHEADER.file_data_vars, RUNHEADER.target_name
    #         )
    #     else:
    #         c_name = "{}{}_Indices_v1.csv".format(
    #             RUNHEADER.file_data_vars, RUNHEADER.target_name
    #         )
    #         assert os.path.isfile(c_name), "Re-assign variables"
    # else:
    #     c_name = None

    # var_names for the target instrument
    c_name = OrderedDict([(int(k), v) for k, v in x_dict.items()])

    # Version 1: using fund raw data (csv)
    (
        dates,
        sd_data,
        y_index_data,
        returns,
        ids_to_class_names,
        ids_to_var_names,
    ) = splite_rawdata_v1(index_price=index_price, y_index=y_index, c_name=c_name)

    (
        dates_new,
        s_test,
        e_test,
        blind_set_seq,
        operation_mode,
    ) = configure_inference_dates(dates, s_test, e_test, forward_ndx)

    # modify data set to reduce time cost
    if _forward_ndx == 20:
        start_ndx = s_test - 175
    elif _forward_ndx == 60:
        start_ndx = s_test - 255
    elif _forward_ndx == 120:
        start_ndx = s_test - 375
    else:
        assert False, "check forward_ndx. the value should be one of [20, 60, 120]"

    # start_ndx = s_test - 370
    end_ndx = e_test
    dates_new = dates_new[start_ndx : end_ndx + 1]
    sd_data = sd_data[start_ndx : end_ndx + 1, :]
    y_index_data = y_index_data[start_ndx : end_ndx + 1, :]
    returns = returns[start_ndx : end_ndx + 1, :]
    s_test = s_test - start_ndx
    e_test = e_test - start_ndx

    class_names_to_ids = dict(
        zip(ids_to_class_names.values(), ids_to_class_names.keys())
    )
    # var_names_to_ids = dict(zip(ids_to_var_names.values(), ids_to_var_names.keys()))

    """Generate re-fined data from raw data
    :param
        input: dates and raw data
    :return
        output: Date aligned raw data
    """

    """declare dataset meta information (part2)
    """
    x_variables = len(sd_data[0])
    # num_y_index = len(y_index_data[0])
    assert x_variables == len(
        ids_to_var_names
    ), "the numbers of x variables are different"

    """Define primitive inputs
        1.price, 2.ratio, 3.velocity
    """
    # calculate statistics for re-fined data
    sd_data = np.array(sd_data, dtype=np.float)
    sd_max = np.max(sd_data, axis=0)
    sd_max = sd_max + sd_max * 0.3  # Buffer
    sd_min = np.min(sd_data, axis=0)
    sd_min = sd_min - sd_min * 0.3  # Buffer

    sd_diff, _, X_unit, _ = trans_val(
        sd_data,
        None,
        ids_to_var_names,
        f_desc=RUNHEADER.var_desc,
        target_name=None,
    )  # daily return
    y_diff = returns

    # sd_diff_max = np.max(sd_diff, axis=0)
    # sd_diff_min = np.min(sd_diff, axis=0)
    # historical observation for a dependency variable
    # historical_ar = y_index_data[:, RUNHEADER.m_target_index]
    # # velocity data
    # sd_velocity = np.diff(sd_diff, axis=0)
    # sd_velocity = np.append([np.zeros(sd_velocity.shape[1])], sd_velocity, axis=0)
    # sd_velocity_max = np.max(sd_velocity, axis=0)
    # sd_velocity_min = np.min(sd_velocity, axis=0)

    """Define inputs
    """
    sd_ma_data_5, sd_ma_data_10, sd_ma_data_20, sd_ma_data_60 = ma(sd_data)
    sd_diff_ma_data_5, sd_diff_ma_data_10, sd_diff_ma_data_20, sd_diff_ma_data_60 = ma(
        sd_diff
    )

    (
        sd_velocity,
        sd_velocity_ma_data_5,
        sd_velocity_ma_data_10,
        sd_velocity_ma_data_20,
        sd_velocity_ma_data_60,
    ) = normalized_spread(
        sd_data, sd_ma_data_5, sd_ma_data_10, sd_ma_data_20, sd_ma_data_60, X_unit
    )

    # (
    #     historical_ar_ma_data_5,
    #     historical_ar_ma_data_10,
    #     historical_ar_ma_data_20,
    #     historical_ar_ma_data_60,
    # ) = (
    #     None,
    #     None,
    #     None,
    #     None,
    # )

    # windowing for extra data
    # fund_his_30 = rolling_apply(fun_cumsum, returns, 30)  # 30days cumulative sum
    # fund_cov_60 = rolling_apply_cov(fun_cov, returns, 60)  # 60days correlation matrix
    # extra_cor_60 = rolling_apply_cov(fun_cov, sd_diff, 60)  # 60days correlation matrix
    # extra_cor_60 = triangular_vector(extra_cor_60)

    # fund_his_30 = None  # 30days cumulative sum
    # fund_cov_60 = None  # 60days correlation matrix
    # extra_cor_60 = None  # 60days correlation matrix
    # extra_cor_60 = None

    mask, _ = get_corr(
        sd_diff, y_diff, X_unit, False, RUNHEADER.m_mask_corr_th
    )  # mask - binary mask
    # update mask with data_var_MANUAL_Indices.csv
    manual = pd.read_csv(f"{RUNHEADER.file_data_vars}MANUAL_Indices.csv", header=None)
    manual_vars = list(manual.values.reshape(-1))
    for k, val in ids_to_var_names.items():
        if val in manual_vars:
            mask[:, k, :] = 1

    # data set split
    sd_dates_train, sd_dates_test = cut_off_data(
        dates_new,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        False,
        forward_ndx,
        ref_forward_ndx,
    )  # dates_new is the data on which conditions(=operation mode) have already been applied
    sd_data_train, sd_data_test = cut_off_data(
        sd_data,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_data_train, sd_diff_data_test = cut_off_data(
        sd_diff,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_data_train, sd_velocity_data_test = cut_off_data(
        sd_velocity,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )

    sd_ma_data_5_train, sd_ma_data_5_test = cut_off_data(
        sd_ma_data_5,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_ma_data_10_train, sd_ma_data_10_test = cut_off_data(
        sd_ma_data_10,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_ma_data_20_train, sd_ma_data_20_test = cut_off_data(
        sd_ma_data_20,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_ma_data_60_train, sd_ma_data_60_test = cut_off_data(
        sd_ma_data_60,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_ma_data_5_train, sd_diff_ma_data_5_test = cut_off_data(
        sd_diff_ma_data_5,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_ma_data_10_train, sd_diff_ma_data_10_test = cut_off_data(
        sd_diff_ma_data_10,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_ma_data_20_train, sd_diff_ma_data_20_test = cut_off_data(
        sd_diff_ma_data_20,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_diff_ma_data_60_train, sd_diff_ma_data_60_test = cut_off_data(
        sd_diff_ma_data_60,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_ma_data_5_train, sd_velocity_ma_data_5_test = cut_off_data(
        sd_velocity_ma_data_5,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_ma_data_10_train, sd_velocity_ma_data_10_test = cut_off_data(
        sd_velocity_ma_data_10,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_ma_data_20_train, sd_velocity_ma_data_20_test = cut_off_data(
        sd_velocity_ma_data_20,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    sd_velocity_ma_data_60_train, sd_velocity_ma_data_60_test = cut_off_data(
        sd_velocity_ma_data_60,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )
    # historical_ar_data_train, historical_ar_data_test = None, None
    # historical_ar_ma_data_5_train, historical_ar_ma_data_5_test = None, None
    # historical_ar_ma_data_10_train, historical_ar_ma_data_10_test = None, None
    # historical_ar_ma_data_20_train, historical_ar_ma_data_20_test = None, None
    # historical_ar_ma_data_60_train, historical_ar_ma_data_60_test = None, None
    # fund_his_30_train, fund_his_30_test = None, None
    # fund_cov_60_train, fund_cov_60_test = None, None
    # extra_cor_60_train, extra_cor_60_test = None, None

    mask_train, mask_test = cut_off_data(
        mask,
        cut_off,
        blind_set_seq,
        s_test,
        e_test,
        operation_mode,
        forward_ndx,
        ref_forward_ndx,
    )

    target_data_train, target_data_test = None, None
    if dependent_var == "returns":
        target_data_train, target_data_test = cut_off_data(
            returns,
            cut_off,
            blind_set_seq,
            s_test,
            e_test,
            operation_mode,
            forward_ndx,
            ref_forward_ndx,
        )
    elif dependent_var == "tri":
        target_data_train, target_data_test = cut_off_data(
            y_index_data,
            cut_off,
            blind_set_seq,
            s_test,
            e_test,
            operation_mode,
            forward_ndx,
            ref_forward_ndx,
        )

    """Write examples
    """
    # verbose description
    TRAIN_WITH_VAL_I = 0
    TEST = 1
    TRAIN_WITHOUT_VAL = 2
    TRAIN_WITH_VAL_D = 3
    TRAIN_WITH_VAL_I_2 = 4

    # # generate the training and validation sets.
    # if verbose is not None:
    #     verbose = int(verbose)
    # _verbose = None

    # if verbose == 0:
    #     _verbose = (
    #         TRAIN_WITH_VAL_I  # general approach - train and validation separately
    #     )
    # elif verbose == 2:  # Train Set configuration
    #     _verbose = TRAIN_WITHOUT_VAL
    # elif verbose == 3:
    #     _verbose = TRAIN_WITH_VAL_D  # duplicated train and validation for early stopping criteria
    # elif verbose == 4:
    #     _verbose = TRAIN_WITH_VAL_I_2  # general approach - train and validation separately with out shard

    return convert_dataset(
        sd_dates_train,
        sd_data_train,
        sd_ma_data_5_train,
        sd_ma_data_10_train,
        sd_ma_data_20_train,
        sd_ma_data_60_train,
        sd_diff_data_train,
        sd_diff_ma_data_5_train,
        sd_diff_ma_data_10_train,
        sd_diff_ma_data_20_train,
        sd_diff_ma_data_60_train,
        sd_velocity_data_train,
        sd_velocity_ma_data_5_train,
        sd_velocity_ma_data_10_train,
        sd_velocity_ma_data_20_train,
        sd_velocity_ma_data_60_train,
        None,
        None,
        None,
        None,
        None,
        target_data_train,
        None,
        None,
        None,
        mask_train,
        x_seq,
        class_names_to_ids,
        None,
        verbose=TRAIN_WITH_VAL_D,
        forward_ndx=forward_ndx,
        ref_forward_ndx=ref_forward_ndx,
        decoder=decoder,
    ), convert_dataset(
        sd_dates_test,
        sd_data_test,
        sd_ma_data_5_test,
        sd_ma_data_10_test,
        sd_ma_data_20_test,
        sd_ma_data_60_test,
        sd_diff_data_test,
        sd_diff_ma_data_5_test,
        sd_diff_ma_data_10_test,
        sd_diff_ma_data_20_test,
        sd_diff_ma_data_60_test,
        sd_velocity_data_test,
        sd_velocity_ma_data_5_test,
        sd_velocity_ma_data_10_test,
        sd_velocity_ma_data_20_test,
        sd_velocity_ma_data_60_test,
        None,
        None,
        None,
        None,
        None,
        target_data_test,
        None,
        None,
        None,
        mask_test,
        x_seq,
        class_names_to_ids,
        None,
        verbose=TEST,
        forward_ndx=forward_ndx,
        ref_forward_ndx=ref_forward_ndx,
        decoder=decoder,
    )
