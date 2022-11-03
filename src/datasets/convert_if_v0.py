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

import header.index_forecasting.RUNHEADER as RUNHEADER
import numpy as np
import pandas as pd
import tensorflow as tf
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
from datasets.x_selection import get_uniqueness


class ReadData(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, date, data, target_data, x_seq, forward_ndx, class_names_to_ids):
        self.source_data = data
        self.target_data = target_data
        self.x_seq = x_seq
        self.forward_ndx = forward_ndx
        self.class_names_to_ids = class_names_to_ids
        self.date = date

    def _get_returns(self, p_data, n_data):
        return ordinary_return(v_init=p_data, v_final=n_data)

    def _get_class_seq(self, data, base_date, forward_ndx, interval):
        tmp = list()
        for days in interval:
            tmp.append(
                self._get_returns(
                    data[base_date, :], data[base_date + forward_ndx + days, :]
                )
            )
        return np.array(tmp, dtype=np.float32)

    # Crop Data
    def _get_patch(self, base_date):
        x_start_ndx = base_date - self.x_seq + 1
        x_end_ndx = base_date + 1
        forward_ndx = self.forward_ndx

        """X data Section
        """
        # given source data
        self.data = self.source_data[
            x_start_ndx:x_end_ndx, :
        ]  # x_seq+1 by the num of variables
        if len(self.data.shape) == 1:
            _ = self.data.shape
        elif len(self.data.shape) == 2:
            self.height, self.width = self.data.shape
        elif len(self.data.shape) == 3:
            _, self.height, self.width = self.data.shape
        else:
            assert False, "None defined dimension!!!"

        # Apply standardization
        std = np.std(self.data, axis=0)
        std = np.where(std == 0, 1e-12, std)
        self.normal_data = (self.data - np.mean(self.data, axis=0) + 1e-12) / std
        assert np.allclose(self.data.shape, self.normal_data.shape)

        # Apply status_data for 5 days (returns)
        previous_data = self.source_data[x_start_ndx - 5 : x_end_ndx - 5, :]
        self.status_data5 = ordinary_return(v_init=previous_data, v_final=self.data)

        # patch min, max
        self.patch_min = np.min(self.data, axis=0)
        self.patch_max = np.max(self.data, axis=0)

        """Y data Section
        """
        # y_seq by the num of variables
        self.class_seq_price = self.target_data[
            base_date + 1 : base_date + forward_ndx + 1, :
        ]
        self.class_seq_height, self.class_seq_width = self.class_seq_price.shape

        self.class_seq_ratio = self._get_class_seq(
            self.target_data, base_date, forward_ndx, [-2, -1, 0, 1, 2]
        )
        self.class_index = self.target_data[
            base_date + forward_ndx, :
        ]  # +20 days Price(index)
        self.base_date_price = self.target_data[base_date, :]  # +0 days Price(index)

        self.class_ratio = self._get_returns(
            self.target_data[base_date, :], self.target_data[base_date + forward_ndx, :]
        )  # +20 days
        self.class_ratio_ref3 = self._get_returns(
            self.target_data[base_date - 1, :], self.target_data[base_date, :]
        )  # today
        self.class_ratio_ref1 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx + ref_forward_ndx[0], :],
        )  # +10 days
        self.class_ratio_ref2 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx + ref_forward_ndx[1], :],
        )  # +15 days
        self.class_ratio_ref4 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx + ref_forward_ndx[2], :],
        )  # +25 days
        self.class_ratio_ref5 = self._get_returns(
            self.target_data[base_date, :],
            self.target_data[base_date + forward_ndx + ref_forward_ndx[3], :],
        )  # +30 days

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
        self.class_label_ref4 = np.where(
            self.class_ratio_ref4 > 0, 1, 0
        )  # +25 days up/down label
        self.class_label_ref5 = np.where(
            self.class_ratio_ref5 > 0, 1, 0
        )  # +30 days up/down label

        """Date data Section
        """
        self.base_date_index = base_date
        self.base_date_label = self.date[base_date]
        self.prediction_date_index = base_date + forward_ndx
        self.prediction_date_label = self.date[base_date + forward_ndx]

    def get_patch(self, base_date):
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

        self.class_seq_height = None
        self.class_seq_width = None
        self.class_seq_price = None
        self.class_seq_ratio = None

        self.class_index = None
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
        self.class_name = None

        self.base_date_price = None
        self.base_date_label = None
        self.base_date_index = None
        self.prediction_date_label = None
        self.prediction_date_index = None

        # extract a patch
        self._get_patch(base_date)


def _get_dataset_filename(dataset_dir, split_name, cv_idx):
    if split_name == "test":
        output_filename = _FILE_PATTERN % (cv_idx, split_name)
    else:
        output_filename = _FILE_PATTERN % (cv_idx, split_name)

    return "{0}/{1}".format(dataset_dir, output_filename)


def cv_index_configuration(date, verbose):
    num_per_shard = int(math.ceil(len(date) / float(_NUM_SHARDS)))
    start_end_index_list = np.zeros([_NUM_SHARDS, 2])  # start and end index
    if verbose == 0:  # train and validation
        for shard_id in range(_NUM_SHARDS):
            start_end_index_list[shard_id] = [
                shard_id * num_per_shard,
                min((shard_id + 1) * num_per_shard, len(date)),
            ]
    else:
        start_end_index_list[0] = [0, len(date)]  # from 0 to end

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
    else:
        index_container = start_end_index_list
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
    sd_min_max_nor_data,
    sd_min_max_nor_ma_data_5,
    sd_min_max_nor_ma_data_10,
    sd_min_max_nor_ma_data_20,
    sd_min_max_nor_ma_data_60,
    target_data,
    fund_his_data_30,
    fund_cov_data_60,
    extra_cov_data_60,
    x_seq,
    forward_ndx,
    class_names_to_ids,
    dataset_dir,
    verbose,
):
    """Converts the given filenames to a TFRecord - tf.train.examples."""

    date = sd_dates

    # Data Binding.. initialize data helper class
    sd_reader = ReadData(
        date, sd_data, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_reader_ma5 = ReadData(
        date, sd_ma_data_5, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_reader_ma10 = ReadData(
        date, sd_ma_data_10, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_reader_ma20 = ReadData(
        date, sd_ma_data_20, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_reader_ma60 = ReadData(
        date, sd_ma_data_60, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_diff_reader = ReadData(
        date, sd_diff_data, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_diff_reader_ma5 = ReadData(
        date, sd_diff_ma_data_5, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_diff_reader_ma10 = ReadData(
        date, sd_diff_ma_data_10, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_diff_reader_ma20 = ReadData(
        date, sd_diff_ma_data_20, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_diff_reader_ma60 = ReadData(
        date, sd_diff_ma_data_60, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_velocity_reader = ReadData(
        date, sd_velocity_data, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_velocity_reader_ma5 = ReadData(
        date, sd_velocity_ma_data_5, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_velocity_reader_ma10 = ReadData(
        date,
        sd_velocity_ma_data_10,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    sd_velocity_reader_ma20 = ReadData(
        date,
        sd_velocity_ma_data_20,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    sd_velocity_reader_ma60 = ReadData(
        date,
        sd_velocity_ma_data_60,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    sd_min_max_nor_reader = ReadData(
        date, sd_min_max_nor_data, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    sd_min_max_nor_reader_ma5 = ReadData(
        date,
        sd_min_max_nor_ma_data_5,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    sd_min_max_nor_reader_ma10 = ReadData(
        date,
        sd_min_max_nor_ma_data_10,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    sd_min_max_nor_reader_ma20 = ReadData(
        date,
        sd_min_max_nor_ma_data_20,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    sd_min_max_nor_reader_ma60 = ReadData(
        date,
        sd_min_max_nor_ma_data_60,
        target_data,
        x_seq,
        forward_ndx,
        class_names_to_ids,
    )
    fund_his_reader_30 = ReadData(
        date, fund_his_data_30, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    fund_cov_reader_60 = ReadData(
        date, fund_cov_data_60, target_data, x_seq, forward_ndx, class_names_to_ids
    )
    extra_cov_reader_60 = ReadData(
        date, extra_cov_data_60, target_data, x_seq, forward_ndx, class_names_to_ids
    )

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
        sd_min_max_nor_reader,
        sd_min_max_nor_reader_ma5,
        sd_min_max_nor_reader_ma10,
        sd_min_max_nor_reader_ma20,
        sd_min_max_nor_reader_ma60,
        fund_his_reader_30,
        fund_cov_reader_60,
        extra_cov_reader_60,
        x_seq,
        forward_ndx,
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
    sd_min_max_nor_reader,
    sd_min_max_nor_reader_ma5,
    sd_min_max_nor_reader_ma10,
    sd_min_max_nor_reader_ma20,
    sd_min_max_nor_reader_ma60,
    fund_his_reader_30,
    fund_cov_reader_60,
    extra_cov_reader_60,
    x_seq,
    forward_ndx,
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
                    sd_min_max_nor_reader,
                    sd_min_max_nor_reader_ma5,
                    sd_min_max_nor_reader_ma10,
                    sd_min_max_nor_reader_ma20,
                    sd_min_max_nor_reader_ma60,
                    fund_his_reader_30,
                    fund_cov_reader_60,
                    extra_cov_reader_60,
                    x_seq,
                    forward_ndx,
                    validation_list,
                    output_filename,
                    stride=1,
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
                    sd_min_max_nor_reader,
                    sd_min_max_nor_reader_ma5,
                    sd_min_max_nor_reader_ma10,
                    sd_min_max_nor_reader_ma20,
                    sd_min_max_nor_reader_ma60,
                    fund_his_reader_30,
                    fund_cov_reader_60,
                    extra_cov_reader_60,
                    x_seq,
                    forward_ndx,
                    train_list,
                    output_filename,
                    stride=2,
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
                sd_min_max_nor_reader,
                sd_min_max_nor_reader_ma5,
                sd_min_max_nor_reader_ma10,
                sd_min_max_nor_reader_ma20,
                sd_min_max_nor_reader_ma60,
                fund_his_reader_30,
                fund_cov_reader_60,
                extra_cov_reader_60,
                x_seq,
                forward_ndx,
                train_list,
                output_filename,
                stride=2,
            )
        else:  # verbose=1 for test
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
                sd_min_max_nor_reader,
                sd_min_max_nor_reader_ma5,
                sd_min_max_nor_reader_ma10,
                sd_min_max_nor_reader_ma20,
                sd_min_max_nor_reader_ma60,
                fund_his_reader_30,
                fund_cov_reader_60,
                extra_cov_reader_60,
                x_seq,
                forward_ndx,
                test_list,
                output_filename,
                stride=1,
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
    sd_min_max_nor_reader,
    sd_min_max_nor_reader_ma5,
    sd_min_max_nor_reader_ma10,
    sd_min_max_nor_reader_ma20,
    sd_min_max_nor_reader_ma60,
    fund_his_reader_30,
    fund_cov_reader_60,
    extra_cov_reader_60,
    x_seq,
    forward_ndx,
    index_container,
    output_filename,
    stride,
):
    # Get patch
    pk_data = list()
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:

        for idx in range(len(index_container)):  # iteration with contained span lists
            start_ndx, end_ndx = index_container[idx]
            start_ndx, end_ndx = int(start_ndx), int(end_ndx)  # type casting
            for i in range(start_ndx, end_ndx, stride):
                sys.stdout.write(
                    "\r>> [%d] Converting data %s" % (idx, output_filename)
                )
                sys.stdout.flush()

                # Read Data
                if ((i - x_seq * 2) > 0) and (
                    (i + forward_ndx + ref_forward_ndx[-1]) < end_ndx
                ):
                    sd_reader.get_patch(i)
                    sd_reader_ma5.get_patch(i)
                    sd_reader_ma10.get_patch(i)
                    sd_reader_ma20.get_patch(i)
                    sd_reader_ma60.get_patch(i)
                    sd_diff_reader.get_patch(i)
                    sd_diff_reader_ma5.get_patch(i)
                    sd_diff_reader_ma10.get_patch(i)
                    sd_diff_reader_ma20.get_patch(i)
                    sd_diff_reader_ma60.get_patch(i)
                    sd_velocity_reader.get_patch(i)
                    sd_velocity_reader_ma5.get_patch(i)
                    sd_velocity_reader_ma10.get_patch(i)
                    sd_velocity_reader_ma20.get_patch(i)
                    sd_velocity_reader_ma60.get_patch(i)
                    sd_min_max_nor_reader.get_patch(i)
                    sd_min_max_nor_reader_ma5.get_patch(i)
                    sd_min_max_nor_reader_ma10.get_patch(i)
                    sd_min_max_nor_reader_ma20.get_patch(i)
                    sd_min_max_nor_reader_ma60.get_patch(i)
                    fund_his_reader_30.get_patch(i)
                    fund_cov_reader_60.get_patch(i)
                    extra_cov_reader_60.get_patch(i)

                    # # general purpose, but need to fix for your modules
                    # example = _tfexample(sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20, sd_reader_ma60,
                    #                      sd_diff_reader, sd_diff_reader_ma5, sd_diff_reader_ma10,
                    #                      sd_diff_reader_ma20, sd_diff_reader_ma60,
                    #                      sd_velocity_reader, sd_velocity_reader_ma5, sd_velocity_reader_ma10,
                    #                      sd_velocity_reader_ma20, sd_velocity_reader_ma60,
                    #                      sd_min_max_nor_reader, sd_min_max_nor_reader_ma5, sd_min_max_nor_reader_ma10,
                    #                      sd_min_max_nor_reader_ma20, sd_min_max_nor_reader_ma60,
                    #                      fund_his_reader_30, fund_cov_reader_60, extra_cov_reader_60)
                    # tfrecord_writer.write(example.SerializeToString())

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
                            sd_min_max_nor_reader,
                            sd_min_max_nor_reader_ma5,
                            sd_min_max_nor_reader_ma10,
                            sd_min_max_nor_reader_ma20,
                            sd_min_max_nor_reader_ma60,
                            fund_his_reader_30,
                            fund_cov_reader_60,
                            extra_cov_reader_60,
                        )
                    )

    pk_output_filename = output_filename.split("tfrecord")[0] + "pkl"
    with open(pk_output_filename, "wb") as fp:
        pickle.dump(pk_data, fp)
        print("\n" + pk_output_filename + ":sample_size " + str(len(pk_data)))
        fp.close()


def _tfexample(
    patch_sd,
    patch_sd_ma5,
    patch_sd_ma10,
    patch_sd_ma20,
    patch_sd_ma60,
    patch_sd_diff,
    patch_sd_diff_ma5,
    patch_sd_diff_ma10,
    patch_sd_diff_ma20,
    patch_sd_diff_ma60,
    patch_sd_velocity,
    patch_sd_velocity_ma5,
    patch_sd_velocity_ma10,
    patch_sd_velocity_ma20,
    patch_sd_velocity_ma60,
    patch_sd_min_max_nor,
    patch_sd_min_max_nor_ma5,
    patch_sd_min_max_nor_ma10,
    patch_sd_min_max_nor_ma20,
    patch_sd_min_max_nor_ma60,
    patch_fund_his_30,
    patch_cov_60,
    patch_extra_cov_reader_60,
):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "structure/data": float_feature(patch_sd.data),
                "structure/data_max": float_feature(sd_max),
                "structure/data_min": float_feature(sd_min),
                "structure/data_ma5": float_feature(patch_sd_ma5.data),
                "structure/data_ma10": float_feature(patch_sd_ma10.data),
                "structure/data_ma20": float_feature(patch_sd_ma20.data),
                "structure/data_ma60": float_feature(patch_sd_ma60.data),
                "structure/normal": float_feature(patch_sd.normal_data),
                "structure/normal_ma5": float_feature(patch_sd_ma5.normal_data),
                "structure/normal_ma10": float_feature(patch_sd_ma10.normal_data),
                "structure/normal_ma20": float_feature(patch_sd_ma20.normal_data),
                "structure/normal_ma60": float_feature(patch_sd_ma60.normal_data),
                "structure/diff": float_feature(patch_sd_diff.normal_data),
                "structure/diff_ma5": float_feature(patch_sd_diff_ma5.normal_data),
                "structure/diff_ma10": float_feature(patch_sd_diff_ma10.normal_data),
                "structure/diff_ma20": float_feature(patch_sd_diff_ma20.normal_data),
                "structure/diff_ma60": float_feature(patch_sd_diff_ma60.normal_data),
                "structure/velocity": float_feature(patch_sd_velocity.normal_data),
                "structure/velocity_ma5": float_feature(
                    patch_sd_velocity_ma5.normal_data
                ),
                "structure/velocity_ma10": float_feature(
                    patch_sd_velocity_ma10.normal_data
                ),
                "structure/velocity_ma20": float_feature(
                    patch_sd_velocity_ma20.normal_data
                ),
                "structure/velocity_ma60": float_feature(
                    patch_sd_velocity_ma60.normal_data
                ),
                "structure/min_max_nor": float_feature(patch_sd_min_max_nor.data),
                "structure/min_max_nor_ma5": float_feature(
                    patch_sd_min_max_nor_ma5.data
                ),
                "structure/min_max_nor_ma10": float_feature(
                    patch_sd_min_max_nor_ma10.data
                ),
                "structure/min_max_nor_ma20": float_feature(
                    patch_sd_min_max_nor_ma20.data
                ),
                "structure/min_max_nor_ma60": float_feature(
                    patch_sd_min_max_nor_ma60.data
                ),
                "structure/extra_cov": float_feature(patch_extra_cov_reader_60.data),
                "structure/height": int64_feature(patch_sd.height),
                "structure/width": int64_feature(patch_sd.width),
                "structure/class/seq_price": float_feature(patch_sd.class_seq_price),
                "structure/class/seq_ratio": float_feature(patch_sd.class_seq_ratio),
                "structure/class/seq_height": int64_feature(patch_sd.class_seq_height),
                "structure/class/seq_width": int64_feature(patch_sd.class_seq_width),
                # 'structure/class/label': int64_feature(patch_sd.class_label),
                "structure/class/index": float_feature(patch_sd.class_index),
                "structure/class/ratio": float_feature(patch_sd.class_ratio),
                "structure/class/ratio_ref0": float_feature(patch_sd.class_ratio_ref0),
                "structure/class/ratio_ref1": float_feature(patch_sd.class_ratio_ref1),
                "structure/class/ratio_ref2": float_feature(patch_sd.class_ratio_ref2),
                "structure/class/ratio_ref3": float_feature(patch_sd.class_ratio_ref3),
                "structure/class/ratio_ref4": float_feature(patch_sd.class_ratio_ref4),
                "structure/class/ratio_ref5": float_feature(patch_sd.class_ratio_ref5),
                "structure/class/ratio_ref6": float_feature(patch_sd.class_ratio_ref6),
                "structure/class/ratio_ref7": float_feature(patch_sd.class_ratio_ref7),
                "fund_his/cov60": float_feature(patch_cov_60.data[-1]),
                "fund_his/data_cumsum30": float_feature(patch_fund_his_30.data[-1]),
                "fund_his/height": int64_feature(patch_fund_his_30.height),
                "fund_his/width": int64_feature(patch_fund_his_30.width),
                "fund_his/patch_min": float_feature(patch_fund_his_30.patch_min),
                "fund_his/patch_max": float_feature(patch_fund_his_30.patch_max),
                "date/base_date_label": bytes_feature(patch_sd.base_date_label),
                "date/base_date_index": int64_feature(patch_sd.base_date_index),
                "date/prediction_date_label": bytes_feature(
                    patch_sd.prediction_date_label
                ),
                "date/prediction_date_index": int64_feature(
                    patch_sd.prediction_date_index
                ),
            }
        )
    )


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


# def get_conjunction_dates_data_v2(sd_dates, y_index_dates, sd_data, y_index_data, returns):
#     sd_dates_true = np.empty(0, dtype=np.int)
#     y_index_dates_true = np.empty(0, dtype=np.int)
#     y_index_dates_true_label = np.empty(0, dtype=np.object)
#
#     print('Validate Working Date!!')
#     for i in range(len(sd_dates)):
#         for k in range(len(y_index_dates)):
#             if sd_dates[i] == y_index_dates[k]:  # conjunction of sd_dates and y_index_dates
#                 if np.sum(np.isnan(y_index_data[k])) > 0:
#                     ValueError('[{}] fund data contains nan'.format(k))
#                 elif np.sum(np.isnan(sd_data[i])) > 0:
#                     ValueError('[{}] index data contains nan'.format(i))
#                 else:
#                     sd_dates_true = np.append(sd_dates_true, i)
#                     y_index_dates_true = np.append(y_index_dates_true, k)
#                     y_index_dates_true_label = np.append(y_index_dates_true_label, y_index_dates[k])
#
#     sd_dates = sd_dates[sd_dates_true]
#     sd_data = sd_data[sd_dates_true]
#
#     y_index_dates = y_index_dates[y_index_dates_true]
#     y_index_data = y_index_data[y_index_dates_true]
#     returns_data = returns[y_index_dates_true]
#
#     assert (len(sd_dates) == len(y_index_dates))
#     assert (len(sd_dates) == len(y_index_data))
#     assert (len(sd_dates) == len(returns_data))
#
#     sd_data = np.array(sd_data, dtype=np.float32)
#     y_index_data = np.array(y_index_data, dtype=np.float32)
#
#     check_nan(sd_data, np.arange(sd_data.shape[1]))
#     check_nan(y_index_data, np.arange(y_index_data.shape[1]))
#
#     return sd_dates, sd_data, y_index_data, returns_data


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


def cut_off_data(data, cut_off, blind_set_seq=None, test_s_date=None, test_e_date=None):
    eof = len(data)

    if test_s_date is None:
        blind_set_seq = eof - blind_set_seq
        if len(data.shape) == 1:  # 1D
            tmp = (
                data[cut_off:blind_set_seq],
                data[blind_set_seq - RUNHEADER.forward_ndx :],
            )
        elif len(data.shape) == 2:  # 2D:
            tmp = (
                data[cut_off:blind_set_seq, :],
                data[blind_set_seq - RUNHEADER.forward_ndx :, :],
            )
        elif len(data.shape) == 3:  # 3D:
            tmp = (
                data[cut_off:blind_set_seq, ::],
                data[blind_set_seq - RUNHEADER.forward_ndx :, :, :],
            )
        else:
            raise IndexError("Define your cut-off code")
    else:
        if len(data.shape) == 1:  # 1D
            tmp = (
                data[cut_off:test_s_date],
                data[test_s_date - RUNHEADER.forward_ndx : test_e_date],
            )
        elif len(data.shape) == 2:  # 2D:
            tmp = (
                data[cut_off:test_s_date, :],
                data[test_s_date - RUNHEADER.forward_ndx : test_e_date :, :],
            )
        elif len(data.shape) == 3:  # 3D:
            tmp = (
                data[cut_off:test_s_date, ::],
                data[test_s_date - RUNHEADER.forward_ndx : test_e_date :, :, :],
            )
        else:
            raise IndexError("Define your cut-off code")
    return tmp


def load_file(file_location, file_format):
    with open(file_location, "rb") as fp:
        if file_format == "npy":
            return np.load(fp)
        elif file_format == "pkl":
            return pickle.load(fp)
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


def data_from_csv(filename, eod=None):
    index_df = pd.read_csv(filename)
    if eod is not None:
        _dates = index_df.values
        e_test_idx = (
            find_date(_dates, eod, -1)
            if len(np.argwhere(_dates == eod)) == 0
            else np.argwhere(_dates == eod)[0][0]
        )
        index_df = index_df.iloc[:e_test_idx, :]

    dates, data = get_working_dates(
        index_df.values[:, 0], np.array(index_df.values[:, 1:], dtype=np.float32)
    )
    ids_to_class_names = dict(zip(range(len(index_df.keys()[1:])), index_df.keys()[1:]))
    return dates, data, ids_to_class_names


def get_data_corresponding(index_price, y_index, eod=None):
    index_dates, index_values, ids_to_var_names = data_from_csv(index_price, eod)
    y_index_dates, y_index_values, ids_to_class_names = data_from_csv(y_index, eod)

    # get working dates
    index_dates, index_values = get_working_dates(index_dates, index_values)
    y_index_dates, y_index_values = get_working_dates(y_index_dates, y_index_values)

    # replace nan with forward fill
    index_values = replace_nan(index_values)

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
    # # update as is
    # if RUNHEADER.gen_var:
    #     get_uniqueness(
    #         file_name=RUNHEADER.raw_x2,
    #         target_name=RUNHEADER.raw_x,
    #         opt="mva",
    #         th=float(RUNHEADER.derived_vars_th[1]),
    #         eod=eod,
    #     )

    # (
    #     dates,
    #     sd_data,
    #     y_index_dates,
    #     y_index_data,
    #     ids_to_var_names,
    #     ids_to_class_names,
    # ) = get_data_corresponding(index_price, y_index)

    (
        dates,
        sd_data,
        y_index_dates,
        y_index_data,
        ids_to_var_names,
        ids_to_class_names,
    ) = get_data_corresponding(
        index_price,
        y_index,
        eod=eod,
    )

    # returns, Caution: this function assume that Y are index or price values
    # returns = ordinary_return(
    #     matrix=y_index_data, unit=current_y_unit(RUNHEADER.target_name)
    # )

    sd_data, ids_to_var_names, opts = gen_pool(
        dates,
        sd_data,
        ids_to_var_names,
        y_index_data[:, RUNHEADER.m_target_index],
        eod=eod,
    )


def _gen_spread(X, Y, ids_to_var_names, num_cov_obs, f_name):
    ids_to_var_names_add = list()
    corr_th = 0.6
    cnt = 0
    idx = 0
    eof = len(ids_to_var_names)

    d_f_summary = pd.read_csv(RUNHEADER.var_desc)
    # f_out = open(f_name + '.csv', 'a')
    while cnt < eof:
        j = cnt + 1
        tmp_dict = {
            "{}-{}".format(ids_to_var_names[cnt], ids_to_var_names[i]): X[:, cnt]
            - X[:, i]
            for i in range(j, eof, 1)
            if unit_datetype.type_check(
                d_f_summary, ids_to_var_names[cnt], ids_to_var_names[i]
            )
        }
        for key, val in tmp_dict.items():
            sys.stdout.write(
                "\r>> [%d/%d] %s matrix calculation....!!!" % (cnt, eof - 1, key)
            )
            sys.stdout.flush()
            cov = rolling_apply_cross_cov(
                fun_cross_cov, val, Y, num_cov_obs
            )  # 60days correlation matrix
            cov = np.where(cov == 1, 0, cov)
            cov = cov[np.argwhere(np.isnan(cov))[-1][0] + 1 :]  # ignore nan

            if len(cov) > 0:
                # print('{}'.format(np.max(np.abs(np.mean(cov, axis=0).squeeze()))), file=f_out)
                _val_test = np.max(np.abs(np.mean(cov, axis=0).squeeze()))
                if (_val_test >= corr_th) and (_val_test < 0.96):
                    print(" extracted key: {}".format(key))
                    ids_to_var_names_add.append([idx, [key, val]])
                    idx = idx + 1
        cnt = cnt + 1
    # print('\n extracted idxs: {}'.format(idx, file=f_out))
    # f_out.close()
    return dict(ids_to_var_names_add)


def gen_spread(data, ids_to_var_names, num_sample_obs, base_first_momentum):
    # Examples
    ma_data = None
    if num_sample_obs is not None:  # whole samples or recent samples
        ma_data = rolling_apply(
            fun_mean,
            data[num_sample_obs[0] : num_sample_obs[1], :],
            base_first_momentum,
        )
    else:
        print("Caution: it contains test samples as well")
        ma_data = rolling_apply(fun_mean, data, base_first_momentum)

    # centering
    scale_v = ma_data[np.argwhere(np.isnan(ma_data))[-1][0] + 1 :]  # ignore nan
    scale_v = RobustScaler().fit_transform(scale_v)
    X_ = scale_v[:, :-1]
    y_ = scale_v[:, -1]

    # nor_diff_add, ids_to_var_names_add = gen_spread_test(X, y, ids_to_var_names, './data_ma')
    # nor_diff_add, ids_to_var_names_add = gen_spread_test(X_, y_, ids_to_var_names, './data_ma_nor')
    num_cov_obs = 60
    return _gen_spread(
        X_,
        y_,
        ids_to_var_names,
        num_cov_obs,
        "./datasets/rawdata/index_data/data_spread_" + RUNHEADER.target_name,
    )


def _pool_adhoc1(data, ids_to_var_names, opt="None", th=0.975):
    return get_uniqueness(
        from_file=False, _data=data, _dict=ids_to_var_names, opt=opt, th=th
    )


def _pool_adhoc2(data, ids_to_var_names):
    return unit_datetype.quantising_vars(data, ids_to_var_names)


def gen_spread_append(
    sd_data,
    target_data,
    ids_to_var_names,
    var_names_to_ids,
    num_sample_obs,
    base_first_momentum,
    eod=None,
):
    # 1. Gen Spread & Append
    data = np.hstack([sd_data, np.expand_dims(target_data, axis=1)])
    if RUNHEADER.disable_derived_vars:
        ids_to_var_names_add = {}
    else:
        ids_to_var_names_add = gen_spread(
            data, ids_to_var_names, num_sample_obs, base_first_momentum
        )
        print("{} variables are added".format(len(ids_to_var_names_add)))
        assert len(ids_to_var_names_add) > 0, "None of variables would be added"

    _dates, _values, _, _, _ids_to_var_names, _ = get_data_corresponding(
        RUNHEADER.raw_x2,
        RUNHEADER.raw_y,
        eod,
    )
    assert (
        sd_data.shape[0] == _values.shape[0]
    ), "target index forces to the un-matching of shapes"

    _var_names_to_ids = dict(
        zip(list(_ids_to_var_names.values()), list(_ids_to_var_names.keys()))
    )

    # 1-1. Append
    c_max_len = len(_ids_to_var_names)
    check_sum = -1
    tmp = [[k, v] for k, v in _ids_to_var_names.items()]
    new_array = list()
    for key, val in ids_to_var_names_add.items():
        check_sum = check_sum + 1
        assert (
            check_sum == key
        ), "Dict keys are not ordered!!! consider OrderedDict instead"

        n_key = c_max_len + key
        tmp.append([n_key, val[0]])
        i_key, j_key = val[0].split("-")
        i_index, j_index = var_names_to_ids[i_key], var_names_to_ids[j_key]

        # # actual index distance
        # new_array.append(data[:, i_index] - data[:, j_index])

        # scaled value distance
        scale_v = np.append(
            np.expand_dims(data[:, i_index], axis=1),
            np.expand_dims(data[:, j_index], axis=1),
            axis=1,
        )
        scale_v = np.hstack([scale_v, np.expand_dims(target_data, axis=1)])
        scale_v = RobustScaler().fit_transform(scale_v)
        new_array.append(scale_v[:, 0] - scale_v[:, 1])

    # 1-2. Re-assign Dictionary and Data
    ids_to_var_names = OrderedDict(tmp)
    var_names_to_ids = OrderedDict(
        zip(list(ids_to_var_names.values()), list(ids_to_var_names.keys()))
    )
    # data = np.append(data[:, :-1], np.array(new_array).T, axis=1)
    if RUNHEADER.disable_derived_vars:
        data = _values
    else:
        data = np.append(_values, np.array(new_array).T, axis=1)

    return data, ids_to_var_names, var_names_to_ids


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
    data = np.hstack([data, np.expand_dims(target_data, axis=1)])

    # 2. pool ordering
    latest_3y_samples = num_sample_obs[1] - (20 * 12 * 3)
    ma_data = rolling_apply(
        fun_mean, data[latest_3y_samples : num_sample_obs[1], :], base_first_momentum
    )

    # cov = rolling_apply_cov(fun_cov, ma_data, num_cov_obs)  # 60days correlation matrix
    # cov = cov[:, :, -1]
    # cov = cov[:, :-1]
    new_cov = np.zeros([ma_data.shape[0], ma_data.shape[1] - 1])
    for idx in range(ma_data.shape[1] - 1):
        sys.stdout.write(
            "\r>> [%d/%d] vectors calculation....!!!" % (idx, ma_data.shape[1] - 1)
        )
        sys.stdout.flush()
        cov = rolling_apply_cross_cov(
            fun_cross_cov, ma_data[:, idx], ma_data[:, -1], num_cov_obs
        )  # 60days correlation matrix
        new_cov[:, idx] = cov[:, 0, 1]
    # 2-1. ordering
    tmp_cov = np.where(np.isnan(new_cov), 0, new_cov)
    # tmp_cov = np.where(tmp_cov == 1, 0, tmp_cov)
    tmp_cov = np.abs(tmp_cov)
    tmp_cov = np.where(tmp_cov >= RUNHEADER.m_pool_corr_th, 1, 0)
    mean_cov = np.nanmean(tmp_cov, axis=0)
    cov_dict = dict(zip(list(ids_to_var_names.values()), mean_cov.tolist()))
    cov_dict = OrderedDict(sorted(cov_dict.items(), key=lambda x: x[1], reverse=True))
    # cov_dict = [[k, v] for k, v in o_cov_dict.items()][::-1]

    # 2-2. Refine
    cov_dict = OrderedDict(
        [[key, val] for key, val in cov_dict.items() if val > explane_th]
    )
    assert len(cov_dict) != 0, "empty list"
    # 2-3. Re-assign Dict & Data
    ordered_ids = [var_names_to_ids[name] for name in cov_dict.keys()]
    # 2-3-1. Apply max_num of variables
    print("the num of variables exceeding explane_th: {}".format(len(ordered_ids)))
    num_variables = len(ordered_ids)
    if num_variables > max_allowed_num_variables:
        ordered_ids = ordered_ids[:max_allowed_num_variables]
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


def gen_pool(dates, sd_data, ids_to_var_names, target_data, eod=None):
    base_first_momentum = 5  # default 5
    # RUNHEADER.m_pool_sample_start = (len(dates) - 750)  # for operation, it has been changed after a experimental
    RUNHEADER.m_pool_sample_start = (
        len(dates) - 70
    )  # for operation, it has been changed after a experimental
    RUNHEADER.m_pool_sample_end = len(dates)
    num_sample_obs = [RUNHEADER.m_pool_sample_start, RUNHEADER.m_pool_sample_end]
    num_cov_obs = 60  # default 60
    max_allowed_num_variables = 8000  # default 5000
    explane_th = RUNHEADER.explane_th
    plot = True  # default False
    opts = None
    var_names_to_ids = dict(
        zip(list(ids_to_var_names.values()), list(ids_to_var_names.keys()))
    )

    def _save(_dates, _data, _ids_to_var_names):
        file_name = RUNHEADER.file_data_vars + RUNHEADER.target_name
        _data = np.append(np.expand_dims(_dates, axis=1), _data, axis=1)
        print("{} saving".format(file_name))
        if RUNHEADER.gen_var:
            _mode = "_intermediate.csv"
        else:
            _mode = ".csv"
            pd.DataFrame(
                data=list(_ids_to_var_names.values()), columns=["VarName"]
            ).to_csv(file_name + "_Indices.csv", index=None, header=None)
            # rewrite
            unit_datetype.script_run(file_name + "_Indices.csv")

        pd.DataFrame(
            data=_data, columns=["TradeDate"] + list(_ids_to_var_names.values())
        ).to_csv(file_name + _mode, index=None)

        print("save done {} ".format(file_name + _mode))
        os._exit(0)

    # 1. Gen Spread & Append
    # 1-1. Append
    # 1-2. Re-assign Dictionary and Data
    gen_spread = RUNHEADER.gen_var
    if gen_spread:
        # # 0. Analysing variables selection
        # sd_data, ids_to_var_names, var_names_to_ids = pool_ordering_refine(sd_data, target_data, ids_to_var_names,
        #                                                                    var_names_to_ids,
        #                                                                    base_first_momentum, num_sample_obs,
        #                                                                    num_cov_obs,
        #                                                                    max_allowed_num_variables, explane_th)
        data, ids_to_var_names, var_names_to_ids = gen_spread_append(
            sd_data,
            target_data,
            ids_to_var_names,
            var_names_to_ids,
            num_sample_obs,
            base_first_momentum,
            eod,
        )
        print("Gen Data shape: {} ".format(data.shape))
        _save(dates, data, ids_to_var_names)
        # file_name = RUNHEADER.file_data_vars + RUNHEADER.target_name
        # data = np.append(np.expand_dims(dates, axis=1), data, axis=1)
        # pd.DataFrame(data=data, columns=['TradeDate'] + list(ids_to_var_names.values())). \
        #     to_csv(file_name + '_intermediate.csv', index=None)
        # exit()
    else:
        # 2. pool ordering
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
        # ad-hoc process
        # filtering with uniqueness
        tmp_data = np.append(np.expand_dims(dates, axis=1), data, axis=1)
        data, ids_to_var_names = _pool_adhoc1(
            tmp_data, ids_to_var_names, opt="mva", th=0.92
        )

        # quantising selected variables
        data, ids_to_var_names = _pool_adhoc2(data, ids_to_var_names)
        assert len(dates) == data.shape[0], "Type Check!!!"
        assert len(ids_to_var_names) == data.shape[1], "Type Check!!!"

        print("Pool Refine Done!!!")
        _save(dates, data, ids_to_var_names)


def ma(data):
    # windowing for sd_data, according to the price
    ma_data_5 = rolling_apply(fun_mean, data, 5)  # 5days moving average
    ma_data_10 = rolling_apply(fun_mean, data, 10)
    ma_data_20 = rolling_apply(fun_mean, data, 20)
    ma_data_60 = rolling_apply(fun_mean, data, 60)
    return ma_data_5, ma_data_10, ma_data_20, ma_data_60


def triangular_vector(data):
    row, n_var, _ = data.shape
    data = data.reshape(row, n_var**2)

    # extract upper-triangular components
    del_idx = list()
    for n_idx in np.arange(n_var):
        if n_idx == 0:
            del_idx.append(0)
        else:
            for n_idx2 in np.arange(n_idx + 1):
                del_idx.append(n_idx * n_var + n_idx2)
    triangular_idx = np.delete(np.arange(n_var**2), del_idx)

    return data[:, triangular_idx]


def run(dataset_dir, file_pattern="fs_v0_cv%02d_%s.tfrecord", s_test=None, e_test=None):
    import header.index_forecasting.RUNHEADER as RUNHEADER

    index_price: str = RUNHEADER.raw_x
    y_index: str = RUNHEADER.raw_y

    splite_rawdata_v1(index_price=index_price, y_index=y_index, eod=e_test)
