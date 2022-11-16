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
import warnings
from collections import OrderedDict

import bottleneck as bn
import numpy as np
import pandas as pd
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
    get_conjunction_dates_data_v3,
    get_manual_vars_additional,
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
from datasets.unit_datetype_des_check import write_var_desc_with_correlation
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

# import tf_slim as slim


def _get_dataset_filename(
    dataset_dir,
    split_name,
    cv_idx,
    file_pattern,
):
    if split_name == "test":
        output_filename = file_pattern % (cv_idx, split_name)
    else:
        output_filename = file_pattern % (cv_idx, split_name)

    return f"{dataset_dir}/{output_filename}"


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
    file_pattern=None,
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

    _convert_dataset(
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
        file_pattern=file_pattern,
    )

    sys.stdout.write("\n")
    sys.stdout.flush()


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
    file_pattern=None,
):
    with tf.Graph().as_default():
        if verbose == 0:  # for train and validation
            for cv_idx, val in enumerate(index_container):
                # validation_list = index_container[cv_idx][0]
                # train_list = index_container[cv_idx][1]
                validation_list = val[0]
                train_list = val[1]
                # for validation
                output_filename = _get_dataset_filename(
                    dataset_dir, "validation", cv_idx, file_pattern
                )
                write_patch(
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
                    output_filename,
                    1,
                    train_sample=False,
                    forward_ndx=forward_ndx,
                    ref_forward_ndx=ref_forward_ndx,
                    decoder=decoder,
                )
                # for train
                output_filename = _get_dataset_filename(
                    dataset_dir, "train", cv_idx, file_pattern
                )
                write_patch(
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
                    train_list,
                    output_filename,
                    2,
                    train_sample=True,
                    forward_ndx=forward_ndx,
                    ref_forward_ndx=ref_forward_ndx,
                    decoder=decoder,
                )
        elif verbose == 2:
            train_list = index_container[[0]]
            # for train only
            output_filename = _get_dataset_filename(
                dataset_dir, "train", 0, file_pattern
            )
            write_patch(
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
                train_list,
                output_filename,
                2,
                train_sample=True,
                forward_ndx=forward_ndx,
                ref_forward_ndx=ref_forward_ndx,
                decoder=decoder,
            )
        elif verbose == 1:  # verbose=1 for test
            test_list = index_container[[0]]
            output_filename = _get_dataset_filename(
                dataset_dir, "test", 0, file_pattern
            )
            write_patch(
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
                output_filename,
                1,
                train_sample=False,
                forward_ndx=forward_ndx,
                ref_forward_ndx=ref_forward_ndx,
                decoder=decoder,
            )
        elif verbose == 3 or verbose == 4:
            validation_list = [index_container[0][0]]
            train_list = [index_container[0][1]]

            # for validation
            output_filename = _get_dataset_filename(
                dataset_dir, "validation", 0, file_pattern
            )
            write_patch(
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
                output_filename,
                1,
                train_sample=False,
                forward_ndx=forward_ndx,
                ref_forward_ndx=ref_forward_ndx,
                decoder=decoder,
            )
            # for train
            output_filename = _get_dataset_filename(
                dataset_dir, "train", 0, file_pattern
            )
            write_patch(
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
                train_list,
                output_filename,
                2,
                train_sample=True,
                forward_ndx=forward_ndx,
                ref_forward_ndx=ref_forward_ndx,
                decoder=decoder,
            )


@funTime("Converting data")
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
    pk_data = []
    data_set_mode = [
        dn for dn in ["test", "train", "validation"] if dn in output_filename
    ][0]

    for idx, val in enumerate(index_container):  # iteration with contained span lists
        start_ndx, end_ndx = val
        start_ndx, end_ndx = int(start_ndx), int(end_ndx)  # type casting
        for i in range(start_ndx, end_ndx, stride):
            # sys.stdout.write(
            #     "\r>> [%d] Converting data %s" % (idx, output_filename)
            # )
            sys.stdout.write(f"\r>> [{idx}] Converting data {output_filename}")
            sys.stdout.flush()

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

    pk_output_filename = output_filename.split("tfrecord")[0] + "pkl"
    with open(pk_output_filename, "wb") as fp:
        pickle.dump(pk_data, fp)
        print(
            f"\n{pk_output_filename}: sample_size {str(len(pk_data))} with stride {stride}"
        )
        fp.close()


# def check_nan(data, keys):
#     check = np.argwhere(np.sum(np.isnan(data), axis=0) == 1)
#     if len(check) > 0:
#         raise ValueError(
#             "{0} contains nan values".format(keys[check.reshape(len(check))])
#         )


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


# def get_read_data(sd_dates, y_index_dates, sd_data, y_index_data):
#     """Validate data and Return actual operation days for target_index"""

#     # 1. [row-wised filter] the conjunction of structure data dates and fund-structure data dates
#     dates, sd_data, y_index_data = get_conjunction_dates_data(
#         sd_dates, y_index_dates, sd_data, y_index_data
#     )

#     # Disable for this data set
#     # # 2. find negative-valued index ..
#     # del_idx = np.argwhere(np.sum(np.where(sd_data < 0, True, False), axis=0) > 0)
#     # if len(del_idx) > 0:
#     #     del_idx = del_idx.reshape(len(del_idx))
#     #
#     # _, all_idx = sd_data.shape
#     # if len(del_idx) > 0:
#     #     positive_value_idx = np.delete(np.arange(all_idx), del_idx)
#     #     sd_data = sd_data[:, positive_value_idx]

#     return dates, sd_data, y_index_data


# def replace_nan(values):
#     return _replace_cond(np.isnan, values)


# def replace_inf(values):
#     return _replace_cond(np.isinf, values)


# def remove_nan(values, target_col=None, axis=0):
#     return _remove_cond(np.isnan, values, target_col=target_col, axis=axis)


# def _get_index_df(v, index_price, ids_to_var_names, target_data=None):
#     x1, x2 = None, None
#     is_exist = False

#     for idx, _ in enumerate(ids_to_var_names):
#         if v == ids_to_var_names[idx]:
#             return index_price[:, idx]

#     if not is_exist:
#         # assert is_exist, "could not find a given variable name: {}".format(v)
#         return np.zeros(index_price.shape[0])


# def _add_vars(index_price, ids_to_var_names, target_data):
#     assert (
#         index_price.shape[0] == target_data.shape[0]
#     ), "the length of the dates for X, Y are different"
#     assert index_price.shape[1] == len(
#         ids_to_var_names
#     ), "the length of X, Dict should be the same"
#     assert target_data.ndim == 1, "Set target index!!!"
#     base_first_momentum, num_cov_obs = 5, 40
#     bin_size = num_cov_obs
#     t1 = index_price[-(bin_size + num_cov_obs) :, :]
#     t2 = target_data[-(bin_size + num_cov_obs) :]

#     t1 = rolling_apply(fun_mean, t1, base_first_momentum)
#     t2 = rolling_apply(fun_mean, t2, base_first_momentum)

#     var_names = list()
#     for idx in range(t1.shape[1]):
#         cov = rolling_apply_cross_cov(fun_cross_cov, t1[:, idx], t2, num_cov_obs)
#         cov = np.where(cov == 1, 0, cov)
#         cov = cov[np.argwhere(np.isnan(cov))[-1][0] + 1 :]  # ignore nan

#         if len(cov) > 0:
#             _val_test = np.max(np.abs(np.mean(cov, axis=0).squeeze()))
#             if (_val_test >= RUNHEADER.m_pool_corr_th) and (_val_test < 0.96):
#                 key = ids_to_var_names[idx]
#                 sys.stdout.write(
#                     "\r>> a extracted key as a additional variable: %s " % (key)
#                 )
#                 sys.stdout.flush()
#                 var_names.append([key, _val_test])

#     alligned_dict = list(
#         OrderedDict(sorted(var_names, key=lambda x: x[1], reverse=True)).keys()
#     )
#     alligned_dict_idx = [
#         key
#         for t_val in alligned_dict
#         for key, val in ids_to_var_names.items()
#         if val == t_val
#     ]

#     _, alligned_dict = get_uniqueness_without_dates(
#         from_file=False,
#         _data=index_price[:, alligned_dict_idx],
#         _dict=alligned_dict,
#         opt="mva",
#     )

#     return alligned_dict


# def get_index_df(index_price=None, ids_to_var_names=None, c_name=None):
#     visit_once = 0
#     for market in list(RUNHEADER.mkidx_mkname.values()):
#         if market in c_name:
#             target_name = market
#             visit_once = visit_once + 1
#             print(f"gather variables for {target_name}")

#             assert visit_once == 1, "not_allow_duplication"

#     c_name = pd.read_csv(c_name, header=None)
#     c_name = c_name.values.squeeze().tolist()

#     if RUNHEADER.re_assign_vars:
#         new_vars = []

#         if RUNHEADER.manual_vars_additional:
#             manual = pd.read_csv(f"{RUNHEADER.file_data_vars}MANUAL_Indices.csv")
#             manual_vars = list(manual.values.reshape(-1))
#             new_vars = c_name + manual_vars
#             new_vars = list(dict.fromkeys(new_vars))

#         c_name = OrderedDict(
#             sorted(zip(new_vars, range(len(new_vars))), key=lambda aa: aa[1])
#         )

#         # save var list
#         file_name = RUNHEADER.file_data_vars + target_name
#         pd.DataFrame(data=list(c_name.keys()), columns=["VarName"]).to_csv(
#             file_name + "_Indices_v1.csv",
#             index=None,
#             header=None,
#         )
#         print(f"{file_name}_Indices_v1.csv has been saved")

#         # save var desc
#         d_f_summary = pd.read_csv(RUNHEADER.var_desc)
#         basename = (file_name + "_Indices_v1.csv").split(".csv")[0]
#         write_var_desc(list(c_name.keys()), d_f_summary, basename)
#     else:
#         c_name = OrderedDict(
#             sorted(zip(c_name, range(len(c_name))), key=lambda aa: aa[1])
#         )

#     index_df = [_get_index_df(v, index_price, ids_to_var_names) for v in c_name.keys()]
#     index_df = np.array(index_df, dtype=np.float32).T

#     # check not in source
#     nis = np.sum(index_df, axis=0) == 0
#     c_nis = np.where(nis == True, False, True)
#     index_df = index_df[:, c_nis]
#     c_name = np.array(list(c_name.keys()))[c_nis].tolist()

#     return np.array(index_df, dtype=np.float32), OrderedDict(
#         sorted(zip(range(len(c_name)), c_name), key=lambda aa: aa[0])
#     )


# def splite_rawdata_v1(index_price=None, y_index=None, c_name=None):
#     index_df = pd.read_csv(index_price)
#     index_df = index_df.ffill(axis=0)
#     index_df = index_df.bfill(axis=0)
#     index_dates = index_df.values[:, 0]
#     index_values = np.array(index_df.values[:, 1:], dtype=np.float32)
#     ids_to_var_names = OrderedDict(
#         zip(range(len(index_df.keys()[1:])), index_df.keys()[1:])
#     )

#     y_index_df = pd.read_csv(y_index)
#     y_index_df = y_index_df.ffill(axis=0)
#     y_index_df = y_index_df.bfill(axis=0)
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
#     # according to the data type of dependent variables, generate return values
#     num_y_var = y_index_data.shape[1]
#     returns = np.zeros(y_index_data.shape)
#     for y_idx in range(num_y_var):
#         target_name = RUNHEADER.target_id2name(y_idx)
#         unit = current_y_unit(target_name)
#         rtn = ordinary_return(matrix=y_index_data, unit=unit)  # daily return
#         returns[:, y_idx] = rtn[:, y_idx]

#     rtn_tuple = (None, None, None, None, None, None)
#     if c_name is not None:
#         for c_name_var in c_name:
#             _sd_data, _ids_to_var_names = get_index_df(
#                 sd_data, ids_to_var_names, c_name_var
#             )

#             if "data_vars_TOTAL_Indices.csv" in c_name_var:
#                 rtn_tuple = (
#                     dates,
#                     copy.deepcopy(_sd_data),
#                     y_index_data,
#                     returns,
#                     ids_to_class_names,
#                     copy.deepcopy(_ids_to_var_names),
#                 )

#     return rtn_tuple


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


def merge2dict(dataset_dir, file_pattern):
    # merge all
    md = [
        it
        for dataset_name in ["train", "validation", "test"]
        for it in load_file(
            _get_dataset_filename(dataset_dir, dataset_name, 0, file_pattern).split(
                "tfrecord"
            )[0]
            + "pkl",
            "pkl",
        )
    ]

    # save
    output_filename = _get_dataset_filename(dataset_dir, "dataset4fv", 0, file_pattern)
    pk_output_filename = output_filename.split("tfrecord")[0] + "pkl"
    with open(pk_output_filename, "wb") as fp:
        pickle.dump(md, fp)
        print("\n" + pk_output_filename + ":sample_size " + str(len(md)))
        fp.close()


def configure_inference_dates(
    operation_mode, dates, s_test=None, e_test=None, forward_ndx=None
):
    blind_set_seq = RUNHEADER.blind_set_seq

    if (
        operation_mode
    ):  # General batch - Add dummy data for inference dates e.g. +20(1M prediction), +60, +120
        #  configure start date
        assert s_test is None, "Base test start dates should be None for operation_mode"
        assert e_test is None, (
            "Base test end dates should be None for operation_mode, "
            "if you want to inference a specific period then, use operation_mode=False"
        )

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
    else:
        if s_test is not None:
            assert not e_test is None, "Base test end dates should not be None"
            blind_set_seq = None
            s_test = (
                find_date(dates, s_test, 1)
                if len(np.argwhere(dates == s_test)) == 0
                else np.argwhere(dates == s_test)[0][0]
            )
            e_test = (
                find_date(dates, e_test, -1)
                if len(np.argwhere(dates == e_test)) == 0
                else np.argwhere(dates == e_test)[0][0]
            )
            # Set Test Date
        else:
            blind_set_seq = RUNHEADER.blind_set_seq
        dates_new = dates
    return dates_new, s_test, e_test, blind_set_seq


def write_variables_info(
    ids_to_class_names, ids_to_var_names, dataset_dir, daily_cov_raw, performed_date
):
    if os.path.isdir(dataset_dir):
        dataset_utils.write_label_file(
            ids_to_class_names, dataset_dir, filename="y_index.txt"
        )
        dataset_utils.write_label_file(
            ids_to_var_names, dataset_dir, filename="x_index.txt"
        )
        dict2json(dataset_dir + "/y_index.json", ids_to_class_names)
        tmp_dict = OrderedDict()
        for key, val in ids_to_var_names.items():
            tmp_dict[str(key)] = val
        dict2json(dataset_dir + "/x_index.json", tmp_dict)

        f_summary = RUNHEADER.var_desc
        t_file = "".join(str(datetime.datetime.now())[:-16].split("-"))

        write_var_desc_with_correlation(
            list(ids_to_var_names.values()),
            daily_cov_raw,
            pd.read_csv(f_summary),
            dataset_dir + f"/x_daily_{t_file}.csv",
            performed_date,
        )
    else:
        ValueError("Dir location does not exist")


def run(
    dataset_dir,
    file_pattern="fs_v0_cv%02d_%s.tfrecord",
    s_test=None,
    e_test=None,
    verbose=2,
    _forward_ndx=None,
    operation_mode=0,
    _performed_date=None,
):
    """Conversion operation.
    Args:
    dataset_dir: The dataset directory where the dataset is stored.
    """
    # index_price = './datasets/rawdata/index_data/prep_index_df_20190821.csv'  # 145 variables
    # index_price = './datasets/rawdata/index_data/01_KOSPI_20190911_176.csv'  # KOSPI
    # index_price = './datasets/rawdata/index_data/KOSPI_20190911_Refine_New_159.csv'  # KOSPI
    # index_price = './datasets/rawdata/index_data/INX_20190909_nonull.csv'  # S&P by jh
    # y_index = './datasets/rawdata/index_data/gold_index.csv'

    index_price = RUNHEADER.raw_x  # S&P by jh
    y_index = RUNHEADER.raw_y
    operation_mode = bool(operation_mode)

    # declare global variables
    # global _FILE_PATTERN
    # _FILE_PATTERN = file_pattern

    ref_forward_ndx = np.array([-10, -5, 5, 10], dtype=np.int)
    ref_forward_ndx = np.array(
        [-int(_forward_ndx * 0.5), -int(_forward_ndx * 0.25), 5, 10], dtype=np.int
    )
    performed_date = _performed_date

    """declare dataset meta information (part1)
    """
    x_seq = 20  # 20days
    forward_ndx = _forward_ndx
    cut_off = 70
    num_of_datatype_obs = 5
    num_of_datatype_obs_total = RUNHEADER.pkexample_type["num_features_1"]

    dependent_var = "tri"
    decoder = globals()[RUNHEADER.pkexample_type["decoder"]]

    c_name = []
    if RUNHEADER.use_c_name:
        for market in list(RUNHEADER.mkidx_mkname.values()):
            file_str = f"{RUNHEADER.file_data_vars}{market}_Indices.csv"
            c_name.append(file_str)
            assert os.path.isfile(
                file_str
            ), f"not exist selected variables for {market}"
    else:
        c_name = None

    # Version 1: get data from csv + manuel variables are involved
    (
        dates,
        sd_data,
        y_index_data,
        returns,
        ids_to_class_names,
        ids_to_var_names,
    ) = splite_rawdata_v1(index_price=index_price, y_index=y_index, c_name=c_name)

    dates_new, s_test, e_test, blind_set_seq = configure_inference_dates(
        operation_mode, dates, s_test, e_test, forward_ndx
    )

    class_names_to_ids = dict(
        zip(ids_to_class_names.values(), ids_to_class_names.keys())
    )
    var_names_to_ids = dict(zip(ids_to_var_names.values(), ids_to_var_names.keys()))

    """declare dataset meta information
    """
    x_variables = len(sd_data[0])
    num_y_index = len(y_index_data[0])
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
    # historical_ar = y_index_data
    # # velocity data - Disable
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

    # Normalized Spread
    # new features - normalized spread
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
    # write file
    write_variables_info(
        ids_to_class_names, ids_to_var_names, dataset_dir, _, performed_date
    )

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
    )
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
    # generate the training and validation sets.
    if verbose is not None:
        verbose = int(verbose)
    _verbose = None

    # verbose description
    TRAIN_WITH_VAL_I = 0
    TEST = 1
    TRAIN_WITHOUT_VAL = 2
    TRAIN_WITH_VAL_D = 3
    TRAIN_WITH_VAL_I_2 = 4

    if verbose == 0:
        _verbose = (
            TRAIN_WITH_VAL_I  # general approach - train and validation separately
        )
    elif verbose == 2:  # Train Set configuration
        _verbose = TRAIN_WITHOUT_VAL
    elif verbose == 3:
        _verbose = TRAIN_WITH_VAL_D  # duplicated train and validation for early stopping criteria
    elif verbose == 4:
        _verbose = TRAIN_WITH_VAL_I_2  # general approach - train and validation separately with out shard

    # Train & Validation
    convert_dataset(
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
        dataset_dir,
        verbose=_verbose,
        forward_ndx=forward_ndx,
        ref_forward_ndx=ref_forward_ndx,
        decoder=decoder,
        file_pattern=file_pattern,
    )

    # Blind set
    convert_dataset(
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
        dataset_dir,
        verbose=TEST,
        forward_ndx=forward_ndx,
        ref_forward_ndx=ref_forward_ndx,
        decoder=decoder,
        file_pattern=file_pattern,
    )

    # Data set to extract feature representation (inspection)
    merge2dict(dataset_dir, file_pattern)

    # write meta information for data set
    meta = {
        "x_seq": x_seq,
        "x_variables": x_variables,
        "forecast": forward_ndx,
        "num_y_index": num_y_index,
        "num_of_datatype_obs": num_of_datatype_obs,
        "num_of_datatype_obs_total": num_of_datatype_obs_total,
        "action_to_y_index": ids_to_class_names,
        "y_index_to_action": class_names_to_ids,
        "idx_to_variable": ids_to_var_names,
        "variable_to_idx": var_names_to_ids,
        "test_set_start": s_test,
        "test_set_end": e_test,
        "verbose": _verbose,
    }
    with open(dataset_dir + "/meta", "wb") as fp:
        pickle.dump(meta, fp)
        fp.close()

    print("\nFinished converting the dataset!")
    print(f"\n Location: {dataset_dir}")
    os._exit(0)
