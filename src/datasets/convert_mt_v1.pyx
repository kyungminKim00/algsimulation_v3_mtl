"""Converts data to TFRecords of TF-Example protos.

This module creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import header.market_timing.RUNHEADER as RUNHEADER
from util import (
    funTime,
    dict2json,
    ordinary_return,
    _replace_cond,
    _remove_cond,
    find_date,
    current_y_unit,
    get_manual_vars_additional,
    trans_val,
)
from datasets.windowing import (
    rolling_apply,
    rolling_apply_cov,
    rolling_apply_cross_cov,
    fun_mean,
    fun_cumsum,
    fun_cov,
    fun_cross_cov,
)
from datasets.x_selection import get_uniqueness, get_uniqueness_without_dates
from datasets.decoder import pkexample_type_A, pkexample_type_B, pkexample_type_C

import math
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from datasets import dataset_utils
import pickle

import datetime
import os
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
from datasets.unit_datetype_des_check import write_var_desc, write_var_desc_with_correlation


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
        # Standardization
        std = np.std(np.array(data, dtype=np.float), axis=0)
        std = np.where(std == 0, 1e-12, std)
        normal_data = (data - np.mean(data, axis=0) + 1e-12) / std
        assert np.allclose(data.shape, normal_data.shape)
        return normal_data

    def _get_williarms(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                _max, _min = np.max(data, axis=0), np.min(data, axis=0)
                wr = (_max - data) / (_max - _min) * -100
                wr = np.where(np.isnan(wr), 0.5, wr)
            except Warning:
                pass
        return wr

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
            self.wiliarms_R = self._get_williarms(self.data)

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
        self.wiliarms_R = None
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


def _get_dataset_filename(dataset_dir, split_name, cv_idx):
    if split_name == "test":
        output_filename = _FILE_PATTERN % (cv_idx, split_name)
    else:
        output_filename = _FILE_PATTERN % (cv_idx, split_name)

    return "{0}/{1}".format(dataset_dir, output_filename)


import tf_slim as slim


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
):
    with tf.Graph().as_default():
        if verbose == 0:  # for train and validation
            for cv_idx in range(len(index_container)):
                validation_list = index_container[cv_idx][0]
                train_list = index_container[cv_idx][1]
                # for validation
                output_filename = _get_dataset_filename(
                    dataset_dir, "validation", cv_idx
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
                    output_filename,
                    stride=1,
                    train_sample=False,
                )
                # for train
                output_filename = _get_dataset_filename(dataset_dir, "train", cv_idx)
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
                    train_list,
                    output_filename,
                    stride=4,
                    train_sample=True,
                )
        elif verbose == 2:
            train_list = index_container[[0]]
            # for train only
            output_filename = _get_dataset_filename(dataset_dir, "train", 0)
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
                train_list,
                output_filename,
                stride=4,
                train_sample=True,
            )
        elif verbose == 1:  # verbose=1 for test
            test_list = index_container[[0]]
            output_filename = _get_dataset_filename(dataset_dir, "test", 0)
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
                output_filename,
                stride=1,
                train_sample=False,
            )
        elif verbose == 3 or verbose == 4:
            validation_list = [index_container[0][0]]
            train_list = [index_container[0][1]]

            # for validation
            output_filename = _get_dataset_filename(dataset_dir, "validation", 0)
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
                output_filename,
                stride=1,
                train_sample=False,
            )
            # for train
            output_filename = _get_dataset_filename(dataset_dir, "train", 0)
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
                train_list,
                output_filename,
                stride=4,
                train_sample=True,
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
):
    # Get patch
    pk_data = list()
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
        data_set_mode = [
            dn for dn in ["test", "train", "validation"] if dn in output_filename
        ][0]

        for idx in range(len(index_container)):  # iteration with contained span lists
            start_ndx, end_ndx = index_container[idx]
            start_ndx, end_ndx = int(start_ndx), int(end_ndx)  # type casting
            for i in range(start_ndx, end_ndx, stride):
                sys.stdout.write(
                    "\r>> [%d] Converting data %s" % (idx, output_filename)
                )
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

    pk_output_filename = output_filename.split("tfrecord")[0] + "pkl"
    with open(pk_output_filename, "wb") as fp:
        pickle.dump(pk_data, fp)
        print("\n" + pk_output_filename + ":sample_size " + str(len(pk_data)))
        fp.close()


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

    # print('RUNHEADER.m_target_index: {}'.format(RUNHEADER.m_target_index))
    # print('len(y_index_data): {}'.format(len(y_index_data)))
    # print('y_index_data.shape: {}'.format(y_index_data.shape))
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

    # Disable for this data set
    # # 2. find negative-valued index ..
    # del_idx = np.argwhere(np.sum(np.where(sd_data < 0, True, False), axis=0) > 0)
    # if len(del_idx) > 0:
    #     del_idx = del_idx.reshape(len(del_idx))
    #
    # _, all_idx = sd_data.shape
    # if len(del_idx) > 0:
    #     positive_value_idx = np.delete(np.arange(all_idx), del_idx)
    #     sd_data = sd_data[:, positive_value_idx]

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


def _add_vars(index_price, ids_to_var_names, target_data):
    assert (
        index_price.shape[0] == target_data.shape[0]
    ), "the length of the dates for X, Y are different"
    assert index_price.shape[1] == len(
        ids_to_var_names
    ), "the length of X, Dict should be the same"
    assert target_data.ndim == 1, "Set target index!!!"
    base_first_momentum, num_cov_obs = 5, 40
    bin_size = num_cov_obs
    t1 = index_price[-(bin_size + num_cov_obs) :, :]
    t2 = target_data[-(bin_size + num_cov_obs) :]

    t1 = rolling_apply(fun_mean, t1, base_first_momentum)
    t2 = rolling_apply(fun_mean, t2, base_first_momentum)

    var_names = list()
    for idx in range(t1.shape[1]):
        cov = rolling_apply_cross_cov(fun_cross_cov, t1[:, idx], t2, num_cov_obs)
        cov = np.where(cov == 1, 0, cov)
        cov = cov[np.argwhere(np.isnan(cov))[-1][0] + 1 :]  # ignore nan

        if len(cov) > 0:
            _val_test = np.max(np.abs(np.mean(cov, axis=0).squeeze()))
            if (_val_test >= RUNHEADER.m_pool_corr_th) and (_val_test < 0.96):
                key = ids_to_var_names[idx]
                sys.stdout.write(
                    "\r>> a extracted key as a additional variable: %s " % (key)
                )
                sys.stdout.flush()
                var_names.append([key, _val_test])

    alligned_dict = list(
        OrderedDict(sorted(var_names, key=lambda x: x[1], reverse=True)).keys()
    )
    alligned_dict_idx = [
        key
        for t_val in alligned_dict
        for key, val in ids_to_var_names.items()
        if val == t_val
    ]

    _, alligned_dict = get_uniqueness_without_dates(
        from_file=False,
        _data=index_price[:, alligned_dict_idx],
        _dict=alligned_dict,
        opt="mva",
    )

    return alligned_dict


def get_index_df(
    index_price=None, ids_to_var_names=None, c_name=None, target_data=None
):

    c_name = pd.read_csv(c_name, header=None)
    c_name = c_name.values.squeeze().tolist()

    def _cross_sampling(*args):
        new_vars = list()
        cnt = 0
        while cnt < len(args):
            cnt = 0
            for arg in args:
                if len(arg) > 0:
                    new_vars.append(arg.pop())
                if not arg:
                    cnt = cnt + 1
        return new_vars

    # add vars to c_name
    sub_file_name = None
    if RUNHEADER.re_assign_vars:
        add_vars = _add_vars(index_price, ids_to_var_names, target_data)
        new_vars = list()

        if RUNHEADER.manual_vars_additional:
            manual_vars = get_manual_vars_additional()
            new_vars = _cross_sampling(
                c_name[::-1], list(add_vars.values())[::-1], manual_vars[::-1]
            )
        else:
            new_vars = _cross_sampling(c_name[::-1], list(add_vars.values())[::-1])
        c_name = OrderedDict(
            sorted(zip(new_vars, range(len(new_vars))), key=lambda aa: aa[1])
        )

        # file out
        if RUNHEADER._debug_on:
            # save var list
            file_name = RUNHEADER.file_data_vars + RUNHEADER.target_name
            pd.DataFrame(data=list(c_name.keys()), columns=["VarName"]).to_csv(
                file_name + "_" + RUNHEADER.dataset_version + "_Indices_v1.csv",
                index=None,
                header=None,
            )
            # save var desc
            f_summary = RUNHEADER.var_desc
            d_f_summary = pd.read_csv(f_summary)
            basename = (
                file_name + "_" + RUNHEADER.dataset_version + "_Indices_v1.csv"
            ).split(".csv")[0]
            write_var_desc(list(c_name.keys()), d_f_summary, basename)
    else:
        c_name = OrderedDict(
            sorted(zip(c_name, range(len(c_name))), key=lambda aa: aa[1])
        )

    index_df = [
        _get_index_df(v, index_price, ids_to_var_names, target_data)
        for v in c_name.keys()
    ]
    index_df = np.array(index_df, dtype=np.float32).T

    # check not in source
    nis = np.sum(index_df, axis=0) == 0
    c_nis = np.where(nis == True, False, True)
    index_df = index_df[:, c_nis]
    c_name = np.array(list(c_name.keys()))[c_nis].tolist()

    return np.array(index_df, dtype=np.float32), OrderedDict(
        sorted(zip(range(len(c_name)), c_name), key=lambda aa: aa[0])
    )


def splite_rawdata_v1(index_price=None, y_index=None, c_name=None):
    def _save(ids_to_var_names):
        file_name = RUNHEADER.file_data_vars + RUNHEADER.target_name
        pd.DataFrame(data=list(ids_to_var_names.values()), columns=["VarName"]).to_csv(
            file_name + "_" + RUNHEADER.dataset_version + "_Indices_v2.csv",
            index=None,
            header=None,
        )
        # save var desc
        f_summary = RUNHEADER.var_desc
        d_f_summary = pd.read_csv(f_summary)
        basename = (
            file_name + "_" + RUNHEADER.dataset_version + "_Indices_v2.csv"
        ).split(".csv")[0]
        write_var_desc(list(ids_to_var_names.values()), d_f_summary, basename)

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

    # dates, sd_data, y_index_data, returns = \
    #     get_conjunction_dates_data_v2(index_dates, y_index_dates, index_values, y_index_values, returns)

    if c_name is not None:
        sd_data, ids_to_var_names = get_index_df(
            sd_data, ids_to_var_names, c_name, y_index_data[:, RUNHEADER.m_target_index]
        )
        # Uniqueness
        tmp_data = np.append(np.expand_dims(dates, axis=1), sd_data, axis=1)
        sd_data, ids_to_var_names = get_uniqueness(
            from_file=False, _data=tmp_data, _dict=ids_to_var_names, opt="mva"
        )

        # file out after uniqueness test
        if RUNHEADER._debug_on:
            _save(ids_to_var_names)

    return dates, sd_data, y_index_data, returns, ids_to_class_names, ids_to_var_names


def ma(data):
    # windowing for sd_data, according to the price
    ma_data_5 = rolling_apply(fun_mean, data, 5)  # 5days moving average
    ma_data_10 = rolling_apply(fun_mean, data, 10)
    ma_data_20 = rolling_apply(fun_mean, data, 20)
    ma_data_60 = rolling_apply(fun_mean, data, 60)
    return ma_data_5, ma_data_10, ma_data_20, ma_data_60


def normalized_spread(data, ma_data_5, ma_data_10, data_20, ma_data_60, X_unit):
    f1, f2, f3, f4, f5 = (
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
        np.zeros(data.shape, dtype=np.float32),
    )
    ma_data_3 = rolling_apply(fun_mean, data, 3)  # 3days moving average

    for idx in range(len(X_unit)):
        f1[:, idx] = ordinary_return(
            v_init=ma_data_3[:, idx], v_final=data[:, idx], unit=X_unit[idx]
        )
        f2[:, idx] = ordinary_return(
            v_init=ma_data_5[:, idx], v_final=data[:, idx], unit=X_unit[idx]
        )
        f3[:, idx] = ordinary_return(
            v_init=ma_data_10[:, idx], v_final=data[:, idx], unit=X_unit[idx]
        )
        f4[:, idx] = ordinary_return(
            v_init=data_20[:, idx], v_final=data[:, idx], unit=X_unit[idx]
        )
        f5[:, idx] = ordinary_return(
            v_init=ma_data_60[:, idx], v_final=data[:, idx], unit=X_unit[idx]
        )

    return f1, f2, f3, f4, f5


def triangular_vector(data):
    row, n_var, _ = data.shape
    data = data.reshape(row, n_var ** 2)

    # extract upper-triangular components
    del_idx = list()
    for n_idx in np.arange(n_var):
        if n_idx == 0:
            del_idx.append(0)
        else:
            for n_idx2 in np.arange(n_idx + 1):
                del_idx.append(n_idx * n_var + n_idx2)
    triangular_idx = np.delete(np.arange(n_var ** 2), del_idx)

    return data[:, triangular_idx]


def _getcorr(data, target_data, base_first_momentum, num_cov_obs, b_scaler=True, opt_mask=None):
    _data = np.hstack([data, np.expand_dims(target_data, axis=1)])
    ma_data = rolling_apply(
        fun_mean, _data, base_first_momentum
    )  # use whole train samples
    cov = rolling_apply_cov(fun_cov, ma_data, num_cov_obs, b_scaler)
    cov = cov[:, :, -1]
    cov = cov[:, :-1]

    tmp_cov = np.where(np.isnan(cov), 0, cov)
    tmp_cov = np.abs(tmp_cov)

    daily_cov_raw = tmp_cov
    tmp_cov = np.where(tmp_cov >= opt_mask, 1, 0)

    return tmp_cov, daily_cov_raw


def get_corr(data, target_data, x_unit=None, y_unit=None, b_scaler=True, opt_mask=None):
    base_first_momentum, num_cov_obs = 5, 40  # default
    tmp_cov, daily_cov_raw = _getcorr(data, target_data, base_first_momentum, num_cov_obs, b_scaler, opt_mask)

    if x_unit is not None:
        add_vol_index = np.array(x_unit) == "volatility"
        tmp_cov = add_vol_index + tmp_cov
        tmp_cov = np.where(tmp_cov >= 1, 1, 0)

    # mean_cov = np.nanmean(tmp_cov, axis=0)
    # cov_dict = dict(zip(list(ids_to_var_names.values()), mean_cov.tolist()))
    # cov_dict = OrderedDict(sorted(cov_dict.items(), key=lambda x: x[1], reverse=True))
    total_num = int(tmp_cov.shape[1] * np.mean(np.mean(tmp_cov)))
    print(
        "the average num of variables on daily: {}".format(total_num)
    )
    if RUNHEADER._debug_on:
        pd.DataFrame(data=tmp_cov).to_csv(
            "{}{}_Cov_{:3.2}.csv".format(
                RUNHEADER.file_data_vars,
                RUNHEADER.target_name,
                RUNHEADER.m_pool_corr_th,
            )
        )
    return tmp_cov, daily_cov_raw


def merge2dict(dataset_dir):
    # merge all
    md = [
        it
        for dataset_name in ["train", "validation", "test"]
        for it in load_file(
            _get_dataset_filename(dataset_dir, dataset_name, 0).split("tfrecord")[0]
            + "pkl",
            "pkl",
        )
    ]

    # Disable - train, validation, and test are duplicated to estimate a state space
    # # remove duplicated data
    # prediction_date_label = [it['date/prediction_date_label'] for it in md]
    # duplicated = [idx for idx in range(len(prediction_date_label)) if
    #               prediction_date_label.count(prediction_date_label[idx]) > 1]
    # duplicated_idx = duplicated[len(duplicated) // 2:]
    # print('Remove duplicated dates: {} duplicated'.format(len(duplicated)))
    #
    # if len(duplicated_idx) > 0:
    #     md = [md[idx] for idx in range(len(md)) if idx not in duplicated_idx]
    # else:
    #     pass

    # save
    output_filename = _get_dataset_filename(dataset_dir, "dataset4fv", 0)
    pk_output_filename = output_filename.split("tfrecord")[0] + "pkl"
    with open(pk_output_filename, "wb") as fp:
        pickle.dump(md, fp)
        print("\n" + pk_output_filename + ":sample_size " + str(len(md)))
        fp.close()


def configure_inference_dates(operation_mode, dates, s_test=None, e_test=None):
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

def write_variables_info(ids_to_class_names, ids_to_var_names, dataset_dir, daily_cov_raw):
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
        write_var_desc_with_correlation(list(ids_to_var_names.values()), daily_cov_raw, pd.read_csv(f_summary), dataset_dir + "/x_daily.csv")
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

    import header.market_timing.RUNHEADER as RUNHEADER

    index_price = RUNHEADER.raw_x  # S&P by jh
    y_index = RUNHEADER.raw_y
    operation_mode = bool(operation_mode)

    # declare global variables
    global sd_max, sd_min, sd_diff_max, sd_diff_min, sd_velocity_max, sd_velocity_min, dependent_var, _NUM_SHARDS, ref_forward_ndx, _FILE_PATTERN, forward_ndx

    _NUM_SHARDS = 5
    _FILE_PATTERN = file_pattern
    ref_forward_ndx = np.array([-10, -5, 5, 10], dtype=np.int)
    ref_forward_ndx = np.array([-int(_forward_ndx*0.5), -int(_forward_ndx*0.25), 5, 10], dtype=np.int)

    """declare dataset meta information (part1)
    """
    x_seq = 20  # 20days
    forward_ndx = _forward_ndx
    cut_off = 70
    num_of_datatype_obs = 5
    num_of_datatype_obs_total = RUNHEADER.pkexample_type["num_features_1"]
    num_of_datatype_obs_total_mt = RUNHEADER.pkexample_type["num_features_2"]
    # RUNHEADER.m_warm_up_4_inference = int(forward_ndx)
    # RUNHEADER.m_warm_up_4_inference = 6

    dependent_var = "tri"
    global g_x_seq, g_num_of_datatype_obs, g_x_variables, g_num_of_datatype_obs_total, g_num_of_datatype_obs_total_mt, decoder

    decoder = globals()[RUNHEADER.pkexample_type["decoder"]]

    # var_names for the target instrument
    if RUNHEADER.use_c_name:
        c_name = "{}{}_Indices.csv".format(
            RUNHEADER.file_data_vars, RUNHEADER.target_name
        )
        assert os.path.isfile(c_name), "Re-assign variables"
    else:
        c_name = None

    # Version 1: using fund raw data (csv)
    (
        dates,
        sd_data,
        y_index_data,
        returns,
        ids_to_class_names,
        ids_to_var_names,
    ) = splite_rawdata_v1(index_price=index_price, y_index=y_index, c_name=c_name)

    dates_new, s_test, e_test, blind_set_seq = configure_inference_dates(
        operation_mode, dates, s_test, e_test
    )
    RUNHEADER.m_pool_sample_start = -(len(dates) - s_test + forward_ndx + 250)
    RUNHEADER.m_pool_sample_end = -(len(dates) - s_test)

    """Todo
    Add variables manualy, Common variables 
    1. search common variables from "ids_to_var_names"
    2. generate add_data and add_ids_to_var_names
    3. mege with blow data and dict
    4. modify RUNHEADER.max_x = RUNHEADER.max_x + the number of additional variables
    """
    if len(ids_to_var_names) > RUNHEADER.max_x:
        sd_data = np.array(sd_data[:, : RUNHEADER.max_x], dtype=np.float)
        ids_to_var_names = OrderedDict(
            list(ids_to_var_names.items())[: RUNHEADER.max_x]
        )

    # # declare global variables
    # global sd_max, sd_min, sd_diff_max, sd_diff_min, sd_velocity_max, sd_velocity_min, dependent_var, \
    #     _NUM_SHARDS, ref_forward_ndx, _FILE_PATTERN, forward_ndx
    #
    # _NUM_SHARDS = 5
    # _FILE_PATTERN = file_pattern
    # ref_forward_ndx = np.array([-10, -5, 5, 10], dtype=np.int)
    #
    # """declare dataset meta information (part1)
    # """
    # x_seq = 20  # 20days
    # forward_ndx = _forward_ndx
    # cut_off = 70
    # num_of_datatype_obs = 5
    # num_of_datatype_obs_total = 15  # 25 -> 15
    #
    # dependent_var = 'tri'
    # global g_x_seq, g_num_of_datatype_obs, g_x_variables, g_num_of_datatype_obs_total, decoder
    # if RUNHEADER.use_var_mask:
    #     decoder = pkexample_type_B
    # else:
    #     decoder = pkexample_type_A

    class_names_to_ids = dict(
        zip(ids_to_class_names.values(), ids_to_class_names.keys())
    )
    var_names_to_ids = dict(zip(ids_to_var_names.values(), ids_to_var_names.keys()))
    # if os.path.isdir(dataset_dir):
    #     dataset_utils.write_label_file(
    #         ids_to_class_names, dataset_dir, filename="y_index.txt"
    #     )
    #     dataset_utils.write_label_file(
    #         ids_to_var_names, dataset_dir, filename="x_index.txt"
    #     )
    #     dict2json(dataset_dir + "/y_index.json", ids_to_class_names)
    #     tmp_dict = OrderedDict()
    #     for key, val in ids_to_var_names.items():
    #         tmp_dict[str(key)] = val
    #     dict2json(dataset_dir + "/x_index.json", tmp_dict)
    # else:
    #     ValueError("Dir location does not exist")

    """Generate re-fined data from raw data
    :param
        input: dates and raw data
    :return
        output: Date aligned raw data
    """
    # # refined raw data from raw data.. Date aligned raw data
    # sd_dates, sd_data, y_index_data = get_read_data(dates, y_index_dates, sd_data, y_index_data)

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
    sd_diff, y_diff, X_unit, Y_unit = trans_val(
        sd_data,
        y_index_data[:, RUNHEADER.m_target_index],
        ids_to_var_names,
        f_desc=RUNHEADER.var_desc,
        target_name=RUNHEADER.target_name,
    )  # daily return
    
    sd_diff_max = np.max(sd_diff, axis=0)
    sd_diff_min = np.min(sd_diff, axis=0)
    # historical observation for a dependency variable
    historical_ar = y_index_data[:, RUNHEADER.m_target_index]
    # # velocity data - Disable
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
    # # Disable
    # (
    #     sd_velocity_ma_data_5,
    #     sd_velocity_ma_data_10,
    #     sd_velocity_ma_data_20,
    #     sd_velocity_ma_data_60,
    # ) = ma(sd_velocity)

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

    (
        historical_ar_ma_data_5,
        historical_ar_ma_data_10,
        historical_ar_ma_data_20,
        historical_ar_ma_data_60,
    ) = ma(historical_ar)

    # windowing for extra data
    fund_his_30 = rolling_apply(fun_cumsum, returns, 30)  # 30days cumulative sum
    fund_cov_60 = rolling_apply_cov(fun_cov, returns, 60)  # 60days correlation matrix
    extra_cor_60 = rolling_apply_cov(fun_cov, sd_diff, 60)  # 60days correlation matrix
    extra_cor_60 = triangular_vector(extra_cor_60)

    mask, daily_cov_raw = get_corr(sd_diff, y_diff, X_unit, Y_unit, False, RUNHEADER.m_mask_corr_th)  # mask - binary mask
    write_variables_info(ids_to_class_names, ids_to_var_names, dataset_dir, daily_cov_raw)
    # mask = get_corr(
    #     sd_data, y_index_data[:, RUNHEADER.m_target_index]
    # )  # mask - binary mask
    print("current idx: {}".format(RUNHEADER.m_target_index))

    # # set cut-off
    # if operation_mode:  # Add dummy data for inference dates e.g. +20(1M prediction), +60, +120
    #     if s_test is not None:
    #         s_test = find_date(dates, s_test, 1) if len(np.argwhere(dates == s_test)) == 0 else \
    #             np.argwhere(dates == s_test)[0][0]
    #     else:  # general case, base date is the latest date on the given data + 1 days
    #         dummy_dates_4_inference = list()
    #         datetime_obj = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
    #         while True:
    #             if len(dummy_dates_4_inference) <= forward_ndx:
    #                 if datetime_obj.weekday() < 5:
    #                     dummy_dates_4_inference.append(datetime_obj.strftime('%Y-%m-%d'))
    #                 else:
    #                     datetime_obj += datetime.timedelta(days=1)
    #             else:
    #                 break
    #         s_test = len(dates) - 1
    #         dates = dates + dummy_dates_4_inference
    # else:
    #     if s_test is not None:
    #         blind_set_seq = None
    #         s_test = find_date(dates, s_test, 1) if len(np.argwhere(dates == s_test)) == 0 else \
    #             np.argwhere(dates == s_test)[0][0]
    #         e_test = find_date(dates, e_test, -1) if len(np.argwhere(dates == e_test)) == 0 else \
    #             np.argwhere(dates == e_test)[0][0]
    #
    #         # Set Test Date
    #     else:
    #         blind_set_seq = RUNHEADER.blind_set_seq

    # added_data = len(dates_new) - len(dates)
    # sd_data = np.concatenate([sd_data, np.random.random([row, len(sd_data.shape[1])])], axis=0)
    # y_index_data = np.concatenate([y_index_data, np.ones([row, len(y_index_data.shape[1])])], axis=0)
    # returns = np.concatenate([returns, np.ones([row, len(returns.shape[1])])], axis=0)

    # data set split
    sd_dates_train, sd_dates_test = cut_off_data(
        dates_new, cut_off, blind_set_seq, s_test, e_test
    )
    sd_data_train, sd_data_test = cut_off_data(
        sd_data, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_diff_data_train, sd_diff_data_test = cut_off_data(
        sd_diff, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    sd_velocity_data_train, sd_velocity_data_test = cut_off_data(
        sd_velocity, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    historical_ar_data_train, historical_ar_data_test = cut_off_data(
        historical_ar, cut_off, blind_set_seq, s_test, e_test, operation_mode
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
    historical_ar_ma_data_5_train, historical_ar_ma_data_5_test = cut_off_data(
        historical_ar_ma_data_5, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    historical_ar_ma_data_10_train, historical_ar_ma_data_10_test = cut_off_data(
        historical_ar_ma_data_10, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    historical_ar_ma_data_20_train, historical_ar_ma_data_20_test = cut_off_data(
        historical_ar_ma_data_20, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    historical_ar_ma_data_60_train, historical_ar_ma_data_60_test = cut_off_data(
        historical_ar_ma_data_60, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    fund_his_30_train, fund_his_30_test = cut_off_data(
        fund_his_30, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    fund_cov_60_train, fund_cov_60_test = cut_off_data(
        fund_cov_60, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
    extra_cor_60_train, extra_cor_60_test = cut_off_data(
        extra_cor_60, cut_off, blind_set_seq, s_test, e_test, operation_mode
    )
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
        historical_ar_data_train,
        historical_ar_ma_data_5_train,
        historical_ar_ma_data_10_train,
        historical_ar_ma_data_20_train,
        historical_ar_ma_data_60_train,
        target_data_train,
        fund_his_30_train,
        fund_cov_60_train,
        extra_cor_60_train,
        mask_train,
        x_seq,
        class_names_to_ids,
        dataset_dir,
        verbose=_verbose,
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
        historical_ar_data_test,
        historical_ar_ma_data_5_test,
        historical_ar_ma_data_10_test,
        historical_ar_ma_data_20_test,
        historical_ar_ma_data_60_test,
        target_data_test,
        fund_his_30_test,
        fund_cov_60_test,
        extra_cor_60_test,
        mask_test,
        x_seq,
        class_names_to_ids,
        dataset_dir,
        verbose=TEST,
    )

    # Data set to extract feature representation (inspection)
    merge2dict(dataset_dir)

    # write meta information for data set
    meta = {
        "x_seq": x_seq,
        "x_variables": x_variables,
        "forecast": forward_ndx,
        "num_y_index": num_y_index,
        "num_of_datatype_obs": g_num_of_datatype_obs,
        "num_of_datatype_obs_total": g_num_of_datatype_obs_total,
        "num_of_datatype_obs_total_mt": g_num_of_datatype_obs_total_mt,
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
    print("\n Location: {0}".format(dataset_dir))
    os._exit(0)
