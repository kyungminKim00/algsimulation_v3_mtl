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
    get_corr,
    ma,
    normalized_spread,
    ordinary_return,
    trans_val,
)

from datasets import dataset_utils
from datasets.decoder import pkexample_type_A, pkexample_type_B
from datasets.unit_datetype_des_check import write_var_desc
from datasets.windowing import (
    fun_cov,
    fun_cross_cov,
    fun_cumsum,
    rolling_apply,
    rolling_apply_cov,
    rolling_apply_cross_cov,
)

# import tf_slim as slim
# slim.variable


class ReadData(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, date, data, target_data, x_seq, class_names_to_ids):
        self.source_data = data
        self.target_data = target_data
        self.x_seq = x_seq
        self.class_names_to_ids = class_names_to_ids
        self.date = date

    def _get_returns(self, p_data, n_data, unit="prc"):
        return ordinary_return(v_init=p_data, v_final=n_data, unit=unit)

    def _get_class_seq(self, data, base_date, interval, unit="prc"):
        tmp = list()
        for days in interval:
            tmp.append(
                self._get_returns(
                    data[base_date, :],
                    data[base_date + forward_ndx + days, :],
                    unit=unit,
                )
            )
        return np.array(tmp, dtype=np.float32)

    def _get_normal(self, data):
        std = np.std(np.array(data, dtype=np.float), axis=0)
        std = np.where(std == 0, 1e-12, std)
        normal_data = (data - np.mean(data, axis=0) + 1e-12) / std
        assert np.allclose(data.shape, normal_data.shape)
        return normal_data

    # Crop Data
    def _get_patch(self, base_date, train_sample=True, historical_y=False):

        x_start_ndx = base_date - self.x_seq + 1
        x_end_ndx = base_date + 1

        """X data Section
        """
        # given source data
        if self.source_data.ndim == 1:
            self.data = self.source_data[x_start_ndx:x_end_ndx]
            _ = self.data.shape
        elif self.source_data.ndim == 2:
            self.data = self.source_data[
                x_start_ndx:x_end_ndx, :
            ]  # x_seq+1 by the num of variables
            self.height, self.width = self.data.shape
        elif self.source_data.ndim == 3:
            self.data = self.source_data[x_start_ndx:x_end_ndx, :, :]
            _, self.height, self.width = self.data.shape
        else:
            assert False, "None defined dimension!!!"

        if historical_y:
            self.normal_data = self._get_normal(self.data)
            # daily return
            previous_data = (
                self.source_data[x_start_ndx - 1 : x_end_ndx - 1]
                if self.source_data.ndim == 1
                else self.source_data[x_start_ndx - 1 : x_end_ndx - 1, :]
            )
            self.data_diff = ordinary_return(
                v_init=previous_data,
                v_final=self.data,
                unit=current_y_unit(RUNHEADER.target_name),
            )
            self.normal_data_diff = self._get_normal(self.data_diff)
            self.data_mu = np.mean(self.data_diff)
            self.data_sigma = np.var(self.data_diff)
            # reshape - extract today statistics
            (
                self.normal_data,
                self.data_diff,
                self.normal_data_diff,
                self.data_mu,
                self.data_sigma,
            ) = (
                np.expand_dims(self.normal_data[-1], axis=0),
                np.expand_dims(self.data_diff[-1], axis=0),
                np.expand_dims(self.normal_data_diff[-1], axis=0),
                np.expand_dims(self.data_mu, axis=0),
                np.expand_dims(self.data_sigma, axis=0),
            )
        else:
            # Apply standardization
            self.normal_data = self._get_normal(self.data)

            # Apply status_data for 5 days (returns)
            previous_data = (
                self.source_data[x_start_ndx - 5 : x_end_ndx - 5]
                if self.source_data.ndim == 1
                else self.source_data[x_start_ndx - 5 : x_end_ndx - 5, :]
            )
            self.status_data5 = ordinary_return(v_init=previous_data, v_final=self.data)

            # patch min, max
            self.patch_min = np.min(self.data, axis=0)
            self.patch_max = np.max(self.data, axis=0)

        """Y data Section
        """
        unit = current_y_unit(RUNHEADER.target_name)

        # y_seq by the num of variables
        self.class_seq_price = self.target_data[
            base_date + 1 : base_date + forward_ndx + 1, :
        ]
        self.class_seq_height, self.class_seq_width = self.class_seq_price.shape

        backward_ndx = 5
        self.tr_class_seq_price_minmaxNor = self.target_data[
            base_date - backward_ndx : base_date + forward_ndx + 1, :
        ]
        self.tr_class_seq_price_minmaxNor = (
            self.tr_class_seq_price_minmaxNor[backward_ndx, :]
            - self.tr_class_seq_price_minmaxNor.min(axis=0)
        ) / (
            self.tr_class_seq_price_minmaxNor.max(axis=0)
            - self.tr_class_seq_price_minmaxNor.min(axis=0)
        )

        self.class_index = self.target_data[
            base_date + forward_ndx, :
        ]  # +20 days Price(index)
        self.tr_class_index = self.tr_class_seq_price_minmaxNor
        self.base_date_price = self.target_data[base_date, :]  # +0 days Price(index)

        self.class_ratio = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx, :],
            unit=unit,
        )  # +20 days
        self.class_ratio_ref3 = self._get_returns(
            self.target_data[base_date - 1, :],
            self.target_data[base_date, :],
            unit=unit,
        )  # today
        self.class_ratio_ref1 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx + ref_forward_ndx[0], :],
            unit=unit,
        )  # +10 days
        self.class_ratio_ref2 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx + ref_forward_ndx[1], :],
            unit=unit,
        )  # +15 days
        self.class_label = np.where(
            self.class_ratio > 0, 1, 0
        )  # +20 days up/down label
        self.class_label_ref1 = np.where(
            self.class_ratio_ref1 > 0, 1, 0
        )  # +10 days up/down label
        self.class_label_ref2 = np.where(
            self.class_ratio_ref2 > 0, 1, 0
        )  # +15 days up/down label
        self.class_label_ref3 = np.where(
            self.class_ratio_ref3 > 0, 1, 0
        )  # +0 days up/down label

        self.tr_class_label_call = np.where(
            self.tr_class_index <= 0.2, 1, 0
        )  # call label
        self.tr_class_label_hold = np.where(
            (self.tr_class_index > 0.2) & (self.tr_class_index < 0.8), 1, 0
        )  # hold label
        self.tr_class_label_put = np.where(
            self.tr_class_index >= 0.8, 1, 0
        )  # put label

        if train_sample:
            self.class_seq_ratio = self._get_class_seq(
                self.target_data, base_date, [-2, -1, 0, 1, 2], unit=unit
            )
            self.class_ratio_ref4 = self._get_returns(
                self.target_data[base_date, :],
                self.target_data[base_date + forward_ndx + ref_forward_ndx[2], :],
            )  # +25 days
            self.class_ratio_ref5 = self._get_returns(
                self.target_data[base_date, :],
                self.target_data[base_date + forward_ndx + ref_forward_ndx[3], :],
            )  # +30 days
            self.class_label_ref4 = np.where(
                self.class_ratio_ref4 > 0, 1, 0
            )  # +25 days up/down label
            self.class_label_ref5 = np.where(
                self.class_ratio_ref5 > 0, 1, 0
            )  # +30 days up/down label
        else:
            self.class_seq_ratio = self._get_class_seq(
                self.target_data, base_date, [-2, -1, 0], unit=unit
            )
            self.class_ratio_ref4 = None
            self.class_ratio_ref5 = None
            self.class_label_ref4 = None  # +25 days up/down label
            self.class_label_ref5 = None  # +30 days up/down label

        """Date data Section
        """
        self.base_date_index = base_date
        self.base_date_label = self.date[base_date]
        self.prediction_date_index = base_date + forward_ndx
        self.prediction_date_label = self.date[base_date + forward_ndx]

    def get_patch(self, base_date, train_sample=True, historical_y=False):
        # initialize variables
        self.data = None
        self.height = None
        self.width = None
        self.normal_data = None
        self.status_data5 = None
        self.status_data5_Y = None
        self.diff_data = None
        self.patch_min = None
        self.patch_max = None
        self.normal_data_diff = None
        self.data_diff = None
        self.data_mu = None
        self.data_sigma = None

        self.class_seq_height = None
        self.class_seq_width = None
        self.class_seq_price = None
        self.tr_class_seq_price_minmaxNor = None
        self.class_seq_ratio = None

        self.class_index = None
        self.tr_class_index = None
        self.class_ratio = None
        self.class_ratio_ref0 = None
        self.class_ratio_ref1 = None
        self.class_ratio_ref2 = None
        self.class_ratio_ref3 = None
        self.class_ratio_ref4 = None
        self.class_ratio_ref5 = None
        self.class_ratio_ref6 = None
        self.class_ratio_ref7 = None

        self.class_label = None
        self.class_label_ref1 = None
        self.class_label_ref2 = None
        self.class_label_ref3 = None
        self.class_label_ref4 = None
        self.class_label_ref5 = None
        self.tr_class_label_call = None
        self.tr_class_label_hold = None
        self.tr_class_label_put = None
        self.class_name = None

        self.base_date_price = None
        self.base_date_label = None
        self.base_date_index = None
        self.prediction_date_label = None
        self.prediction_date_index = None

        # extract a patch
        self._get_patch(base_date, train_sample, historical_y)


def cv_index_configuration(date, verbose):
    num_per_shard = int(math.ceil(len(date) / float(_NUM_SHARDS)))
    start_end_index_list = np.zeros([_NUM_SHARDS, 2])  # start and end index
    if verbose == 0:  # train and validation separately
        for shard_id in range(_NUM_SHARDS):
            start_end_index_list[shard_id] = [
                shard_id * num_per_shard,
                min((shard_id + 1) * num_per_shard, len(date)),
            ]
    elif verbose == 1:  # test
        start_end_index_list[0] = [0, len(date)]
    elif verbose == 2:  # from 0 to end - only train without validation
        start_end_index_list[0] = [0, len(date)]
    elif verbose == 3:  # duplicated validation for early stopping criteria
        headbias_from_y_excluded = forward_ndx + ref_forward_ndx[-1]
        duplicated_samples = (
            -40 - headbias_from_y_excluded - RUNHEADER.m_warm_up_4_inference
        )
        start_end_index_list[0] = [0, len(date)]
        start_end_index_list[1] = [len(date) + duplicated_samples, len(date)]
    elif verbose == 4:  # train and validation separately
        headbias_from_y_excluded = forward_ndx + ref_forward_ndx[-1]
        val_samples = -250 - headbias_from_y_excluded  # val samples 1years
        start_end_index_list[0] = [0, len(date) - val_samples]
        start_end_index_list[1] = [len(date) + val_samples, len(date)]
    return _cv_index_configuration(start_end_index_list, verbose), verbose


def _cv_index_configuration(start_end_index_list, verbose):
    index_container = list()
    validation = list()
    train = list()
    if verbose == 0:  # train and validation
        for idx in range(len(start_end_index_list)):
            for ckeck_idx in range(len(start_end_index_list)):
                if ckeck_idx == idx:
                    validation.append(start_end_index_list[ckeck_idx])
                else:
                    train.append(start_end_index_list[ckeck_idx])
            index_container.append([validation, train])
            validation = list()
            train = list()
    elif verbose == 1:
        index_container = start_end_index_list
    elif verbose == 2:
        index_container = start_end_index_list
    elif (
        verbose == 3 or verbose == 4
    ):  # index_container contains validation and training
        index_container.append([start_end_index_list[1], start_end_index_list[0]])
    return index_container


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
):
    """Converts the given filenames to a TFRecord - tf.train.examples."""

    date = sd_dates

    # Data Binding.. initialize data helper class
    sd_reader = ReadData(date, sd_data, target_data, x_seq, class_names_to_ids)
    sd_reader_ma5 = ReadData(date, sd_ma_data_5, target_data, x_seq, class_names_to_ids)
    sd_reader_ma10 = ReadData(
        date, sd_ma_data_10, target_data, x_seq, class_names_to_ids
    )
    sd_reader_ma20 = ReadData(
        date, sd_ma_data_20, target_data, x_seq, class_names_to_ids
    )
    sd_reader_ma60 = ReadData(
        date, sd_ma_data_60, target_data, x_seq, class_names_to_ids
    )
    sd_diff_reader = ReadData(
        date, sd_diff_data, target_data, x_seq, class_names_to_ids
    )
    sd_diff_reader_ma5 = ReadData(
        date, sd_diff_ma_data_5, target_data, x_seq, class_names_to_ids
    )
    sd_diff_reader_ma10 = ReadData(
        date, sd_diff_ma_data_10, target_data, x_seq, class_names_to_ids
    )
    sd_diff_reader_ma20 = ReadData(
        date, sd_diff_ma_data_20, target_data, x_seq, class_names_to_ids
    )
    sd_diff_reader_ma60 = ReadData(
        date, sd_diff_ma_data_60, target_data, x_seq, class_names_to_ids
    )
    sd_velocity_reader = ReadData(
        date, sd_velocity_data, target_data, x_seq, class_names_to_ids
    )
    sd_velocity_reader_ma5 = ReadData(
        date, sd_velocity_ma_data_5, target_data, x_seq, class_names_to_ids
    )
    sd_velocity_reader_ma10 = ReadData(
        date, sd_velocity_ma_data_10, target_data, x_seq, class_names_to_ids
    )
    sd_velocity_reader_ma20 = ReadData(
        date, sd_velocity_ma_data_20, target_data, x_seq, class_names_to_ids
    )
    sd_velocity_reader_ma60 = ReadData(
        date, sd_velocity_ma_data_60, target_data, x_seq, class_names_to_ids
    )
    historical_ar_reader = ReadData(
        date, historical_ar_data, target_data, x_seq, class_names_to_ids
    )
    historical_ar_reader_ma5 = ReadData(
        date, historical_ar_ma_data_5, target_data, x_seq, class_names_to_ids
    )
    historical_ar_reader_ma10 = ReadData(
        date, historical_ar_ma_data_10, target_data, x_seq, class_names_to_ids
    )
    historical_ar_reader_ma20 = ReadData(
        date, historical_ar_ma_data_20, target_data, x_seq, class_names_to_ids
    )
    historical_ar_reader_ma60 = ReadData(
        date, historical_ar_ma_data_60, target_data, x_seq, class_names_to_ids
    )
    fund_his_reader_30 = ReadData(
        date, fund_his_data_30, target_data, x_seq, class_names_to_ids
    )
    fund_cov_reader_60 = ReadData(
        date, fund_cov_data_60, target_data, x_seq, class_names_to_ids
    )
    extra_cov_reader_60 = ReadData(
        date, extra_cov_data_60, target_data, x_seq, class_names_to_ids
    )
    mask_reader = ReadData(date, mask, target_data, x_seq, class_names_to_ids)

    # Data set configuration - generate cross validation index
    index_container, verbose = cv_index_configuration(date, verbose)

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
                test_list,
                "test",
                stride=1,
                train_sample=False,
            )
        elif verbose == 3 or verbose == 4:
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
                validation_list,
                "validation",
                stride=1,
                train_sample=False,
            )
        else:
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
):
    # Get patch
    pk_data = list()
    data_set_mode = [
        dn for dn in ["test", "train", "validation"] if dn in output_filename
    ][0]

    for idx in range(len(index_container)):  # iteration with contained span lists
        start_ndx, end_ndx = index_container[idx]
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
                historical_ar_reader.get_patch(i, train_sample, True)
                historical_ar_reader_ma5.get_patch(i, train_sample, True)
                historical_ar_reader_ma10.get_patch(i, train_sample, True)
                historical_ar_reader_ma20.get_patch(i, train_sample, True)
                historical_ar_reader_ma60.get_patch(i, train_sample, True)
                fund_his_reader_30.get_patch(i, train_sample)
                fund_cov_reader_60.get_patch(i, train_sample)
                extra_cov_reader_60.get_patch(i, train_sample)
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
                        historical_ar_reader,
                        historical_ar_reader_ma5,
                        historical_ar_reader_ma10,
                        historical_ar_reader_ma20,
                        historical_ar_reader_ma60,
                        fund_his_reader_30,
                        fund_cov_reader_60,
                        extra_cov_reader_60,
                        mask_reader,
                        data_set_mode,
                        RUNHEADER.pkexample_type["num_features_1"],
                        RUNHEADER.pkexample_type["num_features_2"],
                    )
                )
    return pk_data


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

    y_index_data, ref = remove_nan(
        y_index_data, target_col=RUNHEADER.m_target_index, axis=0
    )
    if len(ref) > 0:
        y_index_dates = np.delete(y_index_dates, ref)

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


def get_read_data(sd_dates, y_index_dates, sd_data, y_index_data):
    """Validate data and Return actual operation days for target_index"""

    # 1. [row-wised filter] the conjunction of structure data dates and fund-structure data dates
    dates, sd_data, y_index_data = get_conjunction_dates_data(
        sd_dates, y_index_dates, sd_data, y_index_data
    )

    return dates, sd_data, y_index_data


def add_data_4_operation(data, test_e_date=None):
    n_length = test_e_date - data.shape[0]

    if data.ndim == 1:
        add_data = np.zeros([n_length])
    elif data.ndim == 2:
        add_data = np.zeros([n_length, data.shape[1]])
    elif data.ndim == 3:
        add_data = np.zeros([n_length, data.shape[1], data.shape[2]])
    else:
        assert False, "check dimensions"
    return np.append(data, add_data, axis=0)


def cut_off_data(
    data,
    cut_off,
    blind_set_seq=None,
    test_s_date=None,
    test_e_date=None,
    operation_mode=False,
):
    eof = len(data)
    dummy_date = (
        forward_ndx if forward_ndx > ref_forward_ndx[-1] else ref_forward_ndx[-1]
    )
    dummy_date = dummy_date + 5  # for his return of x (feature)

    if operation_mode:
        data = add_data_4_operation(data, test_e_date)

    if test_s_date is None:
        blind_set_seq = eof - blind_set_seq
        if len(data.shape) == 1:  # 1D
            tmp = (
                data[cut_off:blind_set_seq],
                data[
                    blind_set_seq
                    - forward_ndx
                    - dummy_date
                    - RUNHEADER.m_warm_up_4_inference :
                ],
            )
        elif len(data.shape) == 2:  # 2D:
            tmp = (
                data[cut_off:blind_set_seq, :],
                data[
                    blind_set_seq
                    - forward_ndx
                    - dummy_date
                    - RUNHEADER.m_warm_up_4_inference :,
                    :,
                ],
            )
        elif len(data.shape) == 3:  # 3D:
            tmp = (
                data[cut_off:blind_set_seq, ::],
                data[
                    blind_set_seq
                    - forward_ndx
                    - dummy_date
                    - RUNHEADER.m_warm_up_4_inference :,
                    :,
                    :,
                ],
            )
        else:
            raise IndexError("Define your cut-off code")
    else:
        if test_e_date is None or test_s_date == test_e_date:
            if len(data.shape) == 1:  # 1D
                tmp = (
                    data[cut_off:test_s_date],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference :
                    ],
                )
            elif len(data.shape) == 2:  # 2D:
                tmp = (
                    data[cut_off:test_s_date, :],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference :,
                        :,
                    ],
                )
            elif len(data.shape) == 3:  # 3D:
                tmp = (
                    data[cut_off:test_s_date, ::],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference :,
                        :,
                        :,
                    ],
                )
            else:
                raise IndexError("Define your cut-off code")
        else:  # s_date, e_date are given
            if len(data.shape) == 1:  # 1D
                tmp = (
                    data[cut_off:test_s_date],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference : test_e_date
                    ],
                )
            elif len(data.shape) == 2:  # 2D:
                tmp = (
                    data[cut_off:test_s_date, :],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference : test_e_date,
                        :,
                    ],
                )
            elif len(data.shape) == 3:  # 3D:
                tmp = (
                    data[cut_off:test_s_date, ::],
                    data[
                        test_s_date
                        - forward_ndx
                        - dummy_date
                        - RUNHEADER.m_warm_up_4_inference : test_e_date,
                        :,
                        :,
                    ],
                )
            else:
                raise IndexError("Define your cut-off code")
    return tmp


def load_file(file_location, file_format):
    with open(file_location, "rb") as fp:
        if file_format == "npy":
            data = np.load(fp)
            fp.close()
            return data
        elif file_format == "pkl":
            data = pickle.load(fp)
            fp.close()
            return data
        else:
            raise ValueError("non-support file format")


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


def replace_nan(values):
    return _replace_cond(np.isnan, values)


def replace_inf(values):
    return _replace_cond(np.isinf, values)


def remove_nan(values, target_col=None, axis=0):
    return _remove_cond(np.isnan, values, target_col=target_col, axis=axis)


def _get_index_df(v, index_price, ids_to_var_names, target_data=None):
    x1, x2 = None, None
    is_exist = False
    for idx in range(len(ids_to_var_names)):
        if "-" in v:
            _v = v.split("-")
            if _v[0] == ids_to_var_names[idx]:
                x1 = index_price[:, idx]
            if _v[1] == ids_to_var_names[idx]:
                x2 = index_price[:, idx]
            if (x1 is not None) and (x2 is not None):
                scale_v = np.append(
                    np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1), axis=1
                )
                scale_v = np.hstack([scale_v, np.expand_dims(target_data, axis=1)])
                scale_v = RobustScaler().fit_transform(scale_v)
                # return np.abs(scale_v[:, 0] - scale_v[:, 1])
                return scale_v[:, 0] - scale_v[:, 1]
        else:
            if v == ids_to_var_names[idx]:
                return index_price[:, idx]

    if not is_exist:
        # assert is_exist, "could not find a given variable name: {}".format(v)
        return np.zeros(index_price.shape[0])


def get_index_df(
    index_price=None, ids_to_var_names=None, c_name=None, target_data=None
):
    index_df = [
        _get_index_df(v, index_price, ids_to_var_names, target_data)
        for v in c_name.values()
    ]
    index_df = np.array(index_df, dtype=np.float32).T

    return np.array(index_df, dtype=np.float32), c_name


def splite_rawdata_v1(index_price=None, y_index=None, c_name=None):
    index_df = pd.read_csv(index_price)
    index_dates = index_df.values[:, 0]
    index_values = np.array(index_df.values[:, 1:], dtype=np.float32)
    ids_to_var_names = OrderedDict(
        zip(range(len(index_df.keys()[1:])), index_df.keys()[1:])
    )

    y_index_df = pd.read_csv(y_index)
    y_index_dates = y_index_df.values[:, 0]
    y_index_values = np.array(y_index_df.values[:, 1:], dtype=np.float32)
    ids_to_class_names = OrderedDict(
        zip(range(len(y_index_df.keys()[1:])), y_index_df.keys()[1:])
    )

    # get working dates
    index_dates, index_values = get_working_dates(index_dates, index_values)
    y_index_dates, y_index_values = get_working_dates(y_index_dates, y_index_values)

    # replace nan for independent variables only
    index_values = replace_nan(index_values)

    # the conjunction of target and independent variables
    dates, sd_data, y_index_dates, y_index_data = get_conjunction_dates_data_v3(
        index_dates, y_index_dates, index_values, y_index_values
    )

    unit = current_y_unit(RUNHEADER.target_name)
    returns = ordinary_return(matrix=y_index_data, unit=unit)  # daily return

    sd_data, ids_to_var_names = get_index_df(
        sd_data, ids_to_var_names, c_name, y_index_data[:, RUNHEADER.m_target_index]
    )

    return dates, sd_data, y_index_data, returns, ids_to_class_names, ids_to_var_names


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


def configure_inference_dates(dates, s_test=None, e_test=None):

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
    if domain == "index_forecasting":
        import header.index_forecasting.RUNHEADER as RUNHEADER
    else:
        assert False, "None Defined domain problem"

    index_price = RUNHEADER.raw_x  # S&P by jh
    y_index = RUNHEADER.raw_y
    # operation_mode = bool(operation_mode)

    # # declare global variables
    global sd_max, sd_min, sd_diff_max, sd_diff_min, sd_velocity_max, sd_velocity_min, dependent_var, _NUM_SHARDS, ref_forward_ndx, _FILE_PATTERN, forward_ndx
    _NUM_SHARDS = 5
    # _FILE_PATTERN = file_pattern
    ref_forward_ndx = np.array([-10, -5, 5, 10], dtype=np.int)
    ref_forward_ndx = np.array(
        [-int(_forward_ndx * 0.5), -int(_forward_ndx * 0.25), 5, 10], dtype=np.int
    )

    # declare global variables
    global sd_max, sd_min, sd_diff_max, sd_diff_min, sd_velocity_max, sd_velocity_min, dependent_var, forward_ndx

    """declare dataset meta information (part1)
    """
    x_seq = 20  # 20days
    forward_ndx = _forward_ndx
    cut_off = 70
    num_of_datatype_obs = 5
    num_of_datatype_obs_total = RUNHEADER.pkexample_type["num_features_1"]  # 25 -> 15
    num_of_datatype_obs_total_mt = RUNHEADER.pkexample_type["num_features_2"]
    # RUNHEADER.m_warm_up_4_inference = int(forward_ndx)
    # RUNHEADER.m_warm_up_4_inference = 6

    dependent_var = "tri"
    global g_x_seq, g_num_of_datatype_obs, g_x_variables, g_num_of_datatype_obs_total, g_num_of_datatype_obs_total_mt, decoder

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
    ) = configure_inference_dates(dates, s_test, e_test)

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
    var_names_to_ids = dict(zip(ids_to_var_names.values(), ids_to_var_names.keys()))

    """Generate re-fined data from raw data
    :param
        input: dates and raw data
    :return
        output: Date aligned raw data
    """

    """declare dataset meta information (part2)
    """
    x_variables = len(sd_data[0])
    num_y_index = len(y_index_data[0])
    assert x_variables == len(
        ids_to_var_names
    ), "the numbers of x variables are different"

    # init global variables
    (
        g_x_seq,
        g_num_of_datatype_obs,
        g_x_variables,
        g_num_of_datatype_obs_total,
        g_num_of_datatype_obs_total_mt,
    ) = (
        x_seq,
        num_of_datatype_obs,
        x_variables,
        num_of_datatype_obs_total,
        num_of_datatype_obs_total_mt,
    )

    """Define primitive inputs
        1.price, 2.ratio, 3.velocity
    """
    # calculate statistics for re-fined data
    sd_data = np.array(sd_data, dtype=np.float)
    sd_max = np.max(sd_data, axis=0)
    sd_max = sd_max + sd_max * 0.3  # Buffer
    sd_min = np.min(sd_data, axis=0)
    sd_min = sd_min - sd_min * 0.3  # Buffer
    # differential data
    # sd_diff = ordinary_return(matrix=sd_data)  # daily return
    sd_diff, _, X_unit, _ = trans_val(
        sd_data,
        y_index_data[:, RUNHEADER.m_target_index],
        ids_to_var_names,
        f_desc=RUNHEADER.var_desc,
        target_name=RUNHEADER.target_name,
    )  # daily return
    y_diff = returns

    sd_diff_max = np.max(sd_diff, axis=0)
    sd_diff_min = np.min(sd_diff, axis=0)
    # historical observation for a dependency variable
    historical_ar = y_index_data[:, RUNHEADER.m_target_index]
    # # velocity data
    # sd_velocity = np.diff(sd_diff, axis=0)
    # sd_velocity = np.append([np.zeros(sd_velocity.shape[1])], sd_velocity, axis=0)
    # sd_velocity_max = np.max(sd_velocity, axis=0)
    # sd_velocity_min = np.min(sd_velocity, axis=0)

    """Define inputs
    """
    # according to the price, difference, velocity, performs windowing
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

    (
        historical_ar_ma_data_5,
        historical_ar_ma_data_10,
        historical_ar_ma_data_20,
        historical_ar_ma_data_60,
    ) = (
        None,
        None,
        None,
        None,
    )

    # windowing for extra data
    # fund_his_30 = rolling_apply(fun_cumsum, returns, 30)  # 30days cumulative sum
    # fund_cov_60 = rolling_apply_cov(fun_cov, returns, 60)  # 60days correlation matrix
    # extra_cor_60 = rolling_apply_cov(fun_cov, sd_diff, 60)  # 60days correlation matrix
    # extra_cor_60 = triangular_vector(extra_cor_60)

    fund_his_30 = None  # 30days cumulative sum
    fund_cov_60 = None  # 60days correlation matrix
    extra_cor_60 = None  # 60days correlation matrix
    extra_cor_60 = None

    mask, _ = get_corr(
        sd_diff, y_diff, X_unit, False, RUNHEADER.m_mask_corr_th
    )  # mask - binary mask
    # mask = get_corr(
    #     sd_data, y_index_data[:, RUNHEADER.m_target_index]
    # )  # mask - binary mask

    # data set split
    sd_dates_train, sd_dates_test = cut_off_data(
        dates_new, cut_off, blind_set_seq, s_test, e_test
    )  # dates_new is the data on which conditions(=operation mode) have already been applied
    sd_data_train, sd_data_test = cut_off_data(
        sd_data, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_diff_data_train, sd_diff_data_test = cut_off_data(
        sd_diff, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_velocity_data_train, sd_velocity_data_test = cut_off_data(
        sd_velocity, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )

    sd_ma_data_5_train, sd_ma_data_5_test = cut_off_data(
        sd_ma_data_5, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_ma_data_10_train, sd_ma_data_10_test = cut_off_data(
        sd_ma_data_10, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_ma_data_20_train, sd_ma_data_20_test = cut_off_data(
        sd_ma_data_20, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_ma_data_60_train, sd_ma_data_60_test = cut_off_data(
        sd_ma_data_60, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_diff_ma_data_5_train, sd_diff_ma_data_5_test = cut_off_data(
        sd_diff_ma_data_5, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_diff_ma_data_10_train, sd_diff_ma_data_10_test = cut_off_data(
        sd_diff_ma_data_10, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_diff_ma_data_20_train, sd_diff_ma_data_20_test = cut_off_data(
        sd_diff_ma_data_20, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_diff_ma_data_60_train, sd_diff_ma_data_60_test = cut_off_data(
        sd_diff_ma_data_60, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_velocity_ma_data_5_train, sd_velocity_ma_data_5_test = cut_off_data(
        sd_velocity_ma_data_5, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_velocity_ma_data_10_train, sd_velocity_ma_data_10_test = cut_off_data(
        sd_velocity_ma_data_10, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_velocity_ma_data_20_train, sd_velocity_ma_data_20_test = cut_off_data(
        sd_velocity_ma_data_20, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_velocity_ma_data_60_train, sd_velocity_ma_data_60_test = cut_off_data(
        sd_velocity_ma_data_60, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    historical_ar_data_train, historical_ar_data_test = None, None
    historical_ar_ma_data_5_train, historical_ar_ma_data_5_test = None, None
    historical_ar_ma_data_10_train, historical_ar_ma_data_10_test = None, None
    historical_ar_ma_data_20_train, historical_ar_ma_data_20_test = None, None
    historical_ar_ma_data_60_train, historical_ar_ma_data_60_test = None, None
    fund_his_30_train, fund_his_30_test = None, None
    fund_cov_60_train, fund_cov_60_test = None, None
    extra_cor_60_train, extra_cor_60_test = None, None

    mask_train, mask_test = cut_off_data(
        mask, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )

    target_data_train, target_data_test = None, None
    if dependent_var == "returns":
        target_data_train, target_data_test = cut_off_data(
            returns, cut_off, blind_set_seq, s_test, e_test, operation_mode
        )
    elif dependent_var == "tri":
        target_data_train, target_data_test = cut_off_data(
            y_index_data, cut_off, blind_set_seq, s_test, e_test, operation_mode
        )

    """Write examples
    """
    # # generate the training and validation sets.
    # if verbose is not None:
    #     verbose = int(verbose)
    # _verbose = None

    # verbose description
    TRAIN_WITH_VAL_I = 0
    TEST = 1
    TRAIN_WITHOUT_VAL = 2
    TRAIN_WITH_VAL_D = 3
    TRAIN_WITH_VAL_I_2 = 4

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
    )
