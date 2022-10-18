"""Converts data to TFRecords of TF-Example protos.

This module creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import tensorflow as tf
import numpy as np
from datasets.windowing import rolling_apply, fun_mean, fun_cumsum
from datasets import dataset_utils
from datasets.dataset_utils import float_feature, int64_feature, bytes_feature
import pickle


class ReadData(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, date, data, target_data, x_seq, forward_ndx, class_names_to_ids):
        self.source_data = data
        self.target_data = target_data
        self.x_seq = x_seq
        self.forward_ndx = forward_ndx
        self.class_names_to_ids = class_names_to_ids
        self.date = date

    # Crop Data
    def _get_patch(self, base_date):
        x_start_ndx = base_date - self.x_seq + 1
        x_end_ndx = base_date + 1
        forward_ndx = self.forward_ndx

        """X data Section
        """
        # given source data
        self.data = self.source_data[x_start_ndx:x_end_ndx, :]  # x_seq+1 by the num of variables
        if len(self.data.shape) == 1:
            _ = self.data.shape
        elif len(self.data.shape) == 2:
            self.height, self.width = self.data.shape
        elif len(self.data.shape) == 3:
            _, self.height, self.width = self.data.shape
        else:
            assert False, 'None defined dimension!!!'

        # normalize data (volatility)
        std = np.std(self.data, axis=0)
        std = np.where(std == 0, 1E-12, std)
        self.normal_data = \
            (self.data - np.mean(self.data, axis=0) + 1E-12) / std
        assert np.allclose(self.data.shape, self.normal_data.shape)

        # differential data
        self.diff_data = np.diff(self.source_data[x_start_ndx - 1:x_end_ndx, :], axis=0)
        self.diff_data = np.where(self.diff_data == 0, 1E-12, self.diff_data)
        # cash action [0] causes RuntimeWarning, ignore it
        self.diff_data = self.diff_data / self.source_data[x_start_ndx:x_end_ndx, :]
        assert np.allclose(self.data.shape, self.diff_data.shape)

        # patch min, max
        self.patch_min = np.min(self.data, axis=0)
        self.patch_max = np.max(self.data, axis=0)

        """Y data Section
        """
        # y_seq by the num of variables
        self.class_seq_ratio = self.target_data[base_date + 1:base_date + forward_ndx + 1, :]
        self.class_seq_height, self.class_seq_width = self.class_seq_ratio.shape

        self.class_ratio = self.target_data[base_date + forward_ndx, :]

        self.class_ratio_ref0 = self.target_data[base_date + forward_ndx + ref_forward_ndx[0], :]
        self.class_ratio_ref1 = self.target_data[base_date + forward_ndx + ref_forward_ndx[1], :]
        self.class_ratio_ref2 = self.target_data[base_date + forward_ndx + ref_forward_ndx[2], :]
        self.class_ratio_ref4 = self.target_data[base_date + forward_ndx + ref_forward_ndx[3], :]
        self.class_ratio_ref5 = self.target_data[base_date + forward_ndx + ref_forward_ndx[4], :]
        self.class_ratio_ref6 = self.target_data[base_date + forward_ndx + ref_forward_ndx[5], :]
        self.class_ratio_ref7 = self.target_data[base_date + forward_ndx + ref_forward_ndx[6], :]
        self.class_ratio_ref3 = self.target_data[base_date, :]

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
        self.diff_data = None
        self.patch_min = None
        self.patch_max = None

        self.class_seq_height = None
        self.class_seq_width = None
        # self.class_seq_price = None
        self.class_seq_ratio = None

        # self.class_price = None
        self.class_ratio = None
        self.class_ratio_ref0 = None
        self.class_ratio_ref1 = None
        self.class_ratio_ref2 = None
        self.class_ratio_ref3 = None
        self.class_ratio_ref4 = None
        self.class_ratio_ref5 = None
        self.class_ratio_ref6 = None
        self.class_ratio_ref7 = None

        # self.class_id = None
        # self.class_name = None

        self.base_date_label = None
        self.base_date_index = None
        self.prediction_date_label = None
        self.prediction_date_index = None

        # extract a patch
        self._get_patch(base_date)


def _get_dataset_filename(dataset_dir, split_name, cv_idx):
    if split_name == 'test':
        output_filename = _FILE_PATTERN % (cv_idx, split_name)
    else:
        output_filename = _FILE_PATTERN % (cv_idx, split_name)

    return "{0}/{1}".format(dataset_dir, output_filename)


def cv_index_configuration(date, verbose):
    num_per_shard = int(math.ceil(len(date) / float(_NUM_SHARDS)))
    start_end_index_list = np.zeros([_NUM_SHARDS, 2])  # start and end index
    if verbose == 0:  # train and validation
        for shard_id in range(_NUM_SHARDS):
            start_end_index_list[shard_id] = [shard_id * num_per_shard, min((shard_id + 1) * num_per_shard, len(date))]
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


def convert_dataset(sd_dates,
                    sd_data, sd_ma_data_5, sd_ma_data_10, sd_ma_data_20, sd_ma_data_60,
                    target_data, fund_his_data_30,
                    x_seq, forward_ndx, class_names_to_ids, dataset_dir, verbose):
    """Converts the given filenames to a TFRecord - tf.train.examples.
    """

    date = sd_dates

    # Data Binding.. initialize data helper class
    sd_reader = ReadData(date, sd_data, target_data, x_seq, forward_ndx, class_names_to_ids)
    sd_reader_ma5 = ReadData(date, sd_ma_data_5, target_data, x_seq, forward_ndx, class_names_to_ids)
    sd_reader_ma10 = ReadData(date, sd_ma_data_10, target_data, x_seq, forward_ndx, class_names_to_ids)
    sd_reader_ma20 = ReadData(date, sd_ma_data_20, target_data, x_seq, forward_ndx, class_names_to_ids)
    sd_reader_ma60 = ReadData(date, sd_ma_data_60, target_data, x_seq, forward_ndx, class_names_to_ids)
    fund_his_reader_30 = ReadData(date, fund_his_data_30, target_data, x_seq, forward_ndx, class_names_to_ids)

    # Data set configuration - generate cross validation index
    index_container, verbose = cv_index_configuration(date, verbose)

    _convert_dataset(date, sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20,
                     sd_reader_ma60, fund_his_reader_30, x_seq, forward_ndx, index_container,
                     dataset_dir, verbose)

    sys.stdout.write('\n')
    sys.stdout.flush()


def _convert_dataset(date, sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20, sd_reader_ma60,
                     fund_his_reader_30,
                     x_seq, forward_ndx, index_container, dataset_dir, verbose):
    with tf.Graph().as_default():
        if verbose == 0:  # for train and validation
            for cv_idx in range(len(index_container)):
                validation_list = index_container[cv_idx][0]
                train_list = index_container[cv_idx][1]

                # for validation
                output_filename = _get_dataset_filename(dataset_dir, 'validation', cv_idx)
                write_patch(sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20, sd_reader_ma60,
                            fund_his_reader_30,
                            x_seq, forward_ndx, validation_list, output_filename, verbose)

                # for train
                output_filename = _get_dataset_filename(dataset_dir, 'train', cv_idx)
                write_patch(sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20, sd_reader_ma60,
                            fund_his_reader_30,
                            x_seq, forward_ndx, train_list, output_filename, verbose)
        else:
            test_list = index_container[[0]]
            output_filename = _get_dataset_filename(dataset_dir, 'test', 0)
            write_patch(sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20, sd_reader_ma60,
                        fund_his_reader_30,
                        x_seq, forward_ndx, test_list, output_filename, verbose)


def write_patch(sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20, sd_reader_ma60,
                fund_his_reader_30,
                x_seq, forward_ndx, index_container, output_filename, verbose):
    # Get patch
    pk_data = list()
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:

        for idx in range(len(index_container)):  # iteration with contained span lists
            start_ndx, end_ndx = index_container[idx]
            start_ndx, end_ndx = int(start_ndx), int(end_ndx)  # type casting
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting data %s' % output_filename)
                sys.stdout.flush()

                # Read Data
                if ((i - x_seq) > 0) and ((i + forward_ndx + ref_forward_ndx[-1]) < end_ndx):
                    sd_reader.get_patch(i)
                    sd_reader_ma5.get_patch(i)
                    sd_reader_ma10.get_patch(i)
                    sd_reader_ma20.get_patch(i)
                    sd_reader_ma60.get_patch(i)
                    fund_his_reader_30.get_patch(i)

                    # general purpose
                    example = _tfexample(sd_reader, sd_reader_ma5, sd_reader_ma10,
                                         sd_reader_ma20, sd_reader_ma60, fund_his_reader_30)
                    tfrecord_writer.write(example.SerializeToString())

                    # when only support pickle, e.g. mpi
                    pk_data.append(_pkexample(sd_reader, sd_reader_ma5, sd_reader_ma10,
                                              sd_reader_ma20, sd_reader_ma60, fund_his_reader_30))

    pk_output_filename = output_filename.split('tfrecord')[0] + 'pkl'
    with open(pk_output_filename, 'wb') as fp:
        pickle.dump(pk_data, fp)
        fp.close()


def _tfexample(patch_sd, patch_sd_ma5, patch_sd_ma10, patch_sd_ma20, patch_sd_ma60, patch_fund_his_30):
    return tf.train.Example(features=tf.train.Features(feature={
        'structure/data': float_feature(patch_sd.data),
        'structure/data_max': float_feature(sd_max),
        'structure/data_min': float_feature(sd_min),
        'structure/data_ma5': float_feature(patch_sd_ma5.data),
        'structure/data_ma10': float_feature(patch_sd_ma10.data),
        'structure/data_ma20': float_feature(patch_sd_ma20.data),
        'structure/data_ma60': float_feature(patch_sd_ma60.data),
        'structure/normal': float_feature(patch_sd.normal_data),
        'structure/normal_ma5': float_feature(patch_sd_ma5.normal_data),
        'structure/normal_ma10': float_feature(patch_sd_ma10.normal_data),
        'structure/normal_ma20': float_feature(patch_sd_ma20.normal_data),
        'structure/normal_ma60': float_feature(patch_sd_ma60.normal_data),
        'structure/diff': float_feature(patch_sd.diff_data),
        'structure/diff_ma5': float_feature(patch_sd_ma5.diff_data),
        'structure/diff_ma10': float_feature(patch_sd_ma10.diff_data),
        'structure/diff_ma20': float_feature(patch_sd_ma20.diff_data),
        'structure/diff_ma60': float_feature(patch_sd_ma60.diff_data),
        'structure/height': int64_feature(patch_sd.height),
        'structure/width': int64_feature(patch_sd.width),
        # 'structure/class/seq_price': float_feature(patch_sd.class_seq_price),
        'structure/class/seq_ratio': float_feature(patch_sd.class_seq_ratio),
        'structure/class/seq_height': int64_feature(patch_sd.class_seq_height),
        'structure/class/seq_width': int64_feature(patch_sd.class_seq_width),
        # 'structure/class/label': int64_feature(patch_sd.class_id),
        # 'structure/class/price': float_feature(patch_sd.class_price),
        'structure/class/ratio': float_feature(patch_sd.class_ratio),
        'structure/class/ratio_ref0': float_feature(patch_sd.class_ratio_ref0),
        'structure/class/ratio_ref1': float_feature(patch_sd.class_ratio_ref1),
        'structure/class/ratio_ref2': float_feature(patch_sd.class_ratio_ref2),
        'structure/class/ratio_ref3': float_feature(patch_sd.class_ratio_ref3),
        'structure/class/ratio_ref4': float_feature(patch_sd.class_ratio_ref4),
        'structure/class/ratio_ref5': float_feature(patch_sd.class_ratio_ref5),
        'structure/class/ratio_ref6': float_feature(patch_sd.class_ratio_ref6),
        'structure/class/ratio_ref7': float_feature(patch_sd.class_ratio_ref7),
        'fund_his/data_cumsum30': float_feature(patch_fund_his_30.data[-1]),
        'fund_his/height': int64_feature(patch_fund_his_30.height),
        'fund_his/width': int64_feature(patch_fund_his_30.width),
        'fund_his/patch_min': float_feature(patch_fund_his_30.patch_min),
        'fund_his/patch_max': float_feature(patch_fund_his_30.patch_max),
        'date/base_date_label': bytes_feature(patch_sd.base_date_label),
        'date/base_date_index': int64_feature(patch_sd.base_date_index),
        'date/prediction_date_label': bytes_feature(patch_sd.prediction_date_label),
        'date/prediction_date_index': int64_feature(patch_sd.prediction_date_index),
    }))


def _pkexample(patch_sd, patch_sd_ma5, patch_sd_ma10, patch_sd_ma20, patch_sd_ma60, patch_fund_his_30):
    feature = {
        'structure/data': patch_sd.data,
        'structure/data_max': sd_max,
        'structure/data_min': sd_min,
        'structure/data_ma5': patch_sd_ma5.data,
        'structure/data_ma10': patch_sd_ma10.data,
        'structure/data_ma20': patch_sd_ma20.data,
        'structure/data_ma60': patch_sd_ma60.data,
        'structure/normal': patch_sd.normal_data,
        'structure/normal_ma5': patch_sd_ma5.normal_data,
        'structure/normal_ma10': patch_sd_ma10.normal_data,
        'structure/normal_ma20': patch_sd_ma20.normal_data,
        'structure/normal_ma60': patch_sd_ma60.normal_data,
        'structure/diff': patch_sd.diff_data,
        'structure/diff_ma5': patch_sd_ma5.diff_data,
        'structure/diff_ma10': patch_sd_ma10.diff_data,
        'structure/diff_ma20': patch_sd_ma20.diff_data,
        'structure/diff_ma60': patch_sd_ma60.diff_data,
        'structure/height': patch_sd.height,
        'structure/width': patch_sd.width,
        # this is for speed up calculation on fund_selection_v*.py
        'structure/predefined_observation': np.concatenate((patch_sd.normal_data, patch_sd_ma5.data,
                                                            patch_sd_ma10.data, patch_sd_ma20.data,
                                                            patch_sd_ma60.data), axis=0).\
            reshape((g_num_of_datatype_obs, g_x_seq, g_x_variables)),
        # 'structure/class/seq_price': patch_sd.class_seq_price,
        'structure/class/seq_ratio': patch_sd.class_seq_ratio,
        'structure/class/seq_height': patch_sd.class_seq_height,
        'structure/class/seq_width': patch_sd.class_seq_width,
        # 'structure/class/label': patch_sd.class_id,
        # 'structure/class/price': patch_sd.class_price,
        'structure/class/ratio': patch_sd.class_ratio,
        'structure/class/ratio_ref0': patch_sd.class_ratio_ref0,
        'structure/class/ratio_ref1': patch_sd.class_ratio_ref1,
        'structure/class/ratio_ref2': patch_sd.class_ratio_ref2,
        'structure/class/ratio_ref3': patch_sd.class_ratio_ref3,
        'structure/class/ratio_ref4': patch_sd.class_ratio_ref4,
        'structure/class/ratio_ref5': patch_sd.class_ratio_ref5,
        'structure/class/ratio_ref6': patch_sd.class_ratio_ref6,
        'structure/class/ratio_ref7': patch_sd.class_ratio_ref7,
        'fund_his/data_cumsum30': patch_fund_his_30.data[-1],
        'fund_his/height': patch_fund_his_30.height,
        'fund_his/width': patch_fund_his_30.width,
        'fund_his/patch_min': patch_fund_his_30.patch_min,
        'fund_his/patch_max': patch_fund_his_30.patch_max,
        'date/base_date_label': patch_sd.base_date_label,
        'date/base_date_index': patch_sd.base_date_index,
        'date/prediction_date_label': patch_sd.prediction_date_label,
        'date/prediction_date_index': patch_sd.prediction_date_index,
    }
    return feature


def check_nan(data, keys):
    check = np.argwhere(np.sum(np.isnan(data), axis=0) == 1)
    if len(check) > 0:
        raise ValueError('{0} contains nan values'.format(keys[check.reshape(len(check))]))


def get_conjunction_dates_data(sd_dates, fund_dates, sd_data, fund_data):
    sd_dates_true = np.empty(0, dtype=np.int)
    fund_dates_true = np.empty(0, dtype=np.int)
    fund_dates_true_label = np.empty(0, dtype=np.object)

    for i in range(len(sd_dates)):
        for k in range(len(fund_dates)):
            if sd_dates[i] == fund_dates[k]:  # conjunction of sd_dates and fund_dates
                if np.sum(np.isnan(fund_data[:, 0])) == 0:
                    sd_dates_true = np.append(sd_dates_true, i)
                    fund_dates_true = np.append(fund_dates_true, k)
                    fund_dates_true_label = np.append(fund_dates_true_label, fund_dates[k])

    sd_dates = sd_dates[sd_dates_true]
    sd_data = sd_data[sd_dates_true]

    fund_dates = fund_dates[fund_dates_true]

    assert (len(sd_dates) == len(fund_dates))
    assert (len(sd_dates) == len(fund_data))
    check_nan(sd_data, np.arange(sd_data.shape[1]))
    check_nan(fund_data, np.arange(fund_data.shape[1]))

    return sd_dates, sd_data, fund_data


def get_read_data(sd_dates, fund_dates, sd_data, fund_data):
    """Validate data and Return actual operation days for target_index
    """

    # 1. [row-wised filter] the conjunction of structure data dates and fund-structure data dates
    dates, sd_data, fund_data = get_conjunction_dates_data(sd_dates, fund_dates, sd_data, fund_data)

    # 2. find negative-valued index
    del_idx = np.argwhere(np.sum(np.where(sd_data < 0, True, False), axis=0) > 0)
    if len(del_idx) > 0:
        del_idx = del_idx.reshape(len(del_idx))

    _, all_idx = sd_data.shape
    if len(del_idx) > 0:
        positive_value_idx = np.delete(np.arange(all_idx), del_idx)
        sd_data = sd_data[:, positive_value_idx]

    return dates, sd_data, fund_data


def cut_off_data(data, cut_off, blind_set_seq):
    eof = len(data)
    blind_set_seq = eof - blind_set_seq

    if len(data.shape) == 1:  # 1D
        tmp = data[cut_off:blind_set_seq], data[blind_set_seq:]
    elif len(data.shape) == 2:  # 2D:
        tmp = data[cut_off:blind_set_seq, :], data[blind_set_seq:, :]
    elif len(data.shape) == 3:  # 3D:
        tmp = data[cut_off:blind_set_seq, ::], data[blind_set_seq:, :, :]
    else:
        raise IndexError('Define your cut-off code')
    return tmp


def load_file(file_location, file_format):
    with open(file_location, 'rb') as fp:
        if file_format == 'npy':
            return np.load(fp)
        elif file_format == 'pkl':
            return pickle.load(fp)
        else:
            raise ValueError('non-support file format')


def run(dataset_dir, file_pattern='fs_v1_cv%02d_%s.tfrecord'):
    """Conversion operation.
        Args:
        dataset_dir: The dataset directory where the dataset is stored.
    """
    dates = load_file('./datasets/rawdata/fund_data/dates.npy', 'npy')
    ids_to_class_names = load_file('./datasets/rawdata/fund_data/fund_id.pkl', 'pkl')
    returns = load_file('./datasets/rawdata/fund_data/returns.npy', 'npy')
    sd_data = load_file('./datasets/rawdata/fund_data/sdata.npy', 'npy')
    log_returns = load_file('./datasets/rawdata/fund_data/log_returns.npy', 'npy')
    fund_data = load_file('./datasets/rawdata/fund_data/tri.npy', 'npy')
    fund__dates = load_file('./datasets/rawdata/fund_data/tri_dates.npy', 'npy')

    global sd_max, sd_min, _NUM_SHARDS, ref_forward_ndx, _FILE_PATTERN
    _NUM_SHARDS = 5
    _FILE_PATTERN = file_pattern
    ref_forward_ndx = np.array([-4, 5, 15, 25, 35, 45, 55], dtype=np.int)

    """declare dataset meta information (part1)
    """
    x_seq = 20  # 20days
    forward_ndx = 5  # the days after 5 days
    blind_set_seq = 500
    cut_off = 70
    num_of_datatype_obs = 5
    dependent_var = 'returns'
    global g_x_seq, g_num_of_datatype_obs, g_x_variables

    # fix data index (유진이가 준 수익률 데이터 오늘의 수익률을 하루 밀려서 계산함)
    returns = np.append([np.zeros(returns.shape[1])], returns, axis=0)
    returns = returns[:-1, :] * 100  # 수익률 %로 변환
    log_returns = np.append([np.zeros(log_returns.shape[1])], log_returns, axis=0)
    log_returns = log_returns[:-1, :]

    class_names_to_ids = dict(zip(ids_to_class_names.values(), ids_to_class_names.keys()))
    dataset_utils.write_label_file(ids_to_class_names, dataset_dir, filename='actions.txt')

    """Generate re-fined data from raw data
    :param
        input: dates and raw data
    :return
        output: Date aligned raw data
    """
    # refined raw data from raw data.. Date aligned raw data
    sd_dates, sd_data, fund_data = get_read_data(dates, fund__dates, sd_data, fund_data)

    """declare dataset meta information (part2)
    """
    # Todo: check it later, 데이터 정말 이상함, 우선 임으로 삭제
    x_variables = len(sd_data[0])
    num_fund = len(fund_data[0])

    # init global variables
    g_x_seq, g_num_of_datatype_obs, g_x_variables = x_seq, num_of_datatype_obs, x_variables

    # calculate statistics for re-fined data
    sd_max = np.max(sd_data, axis=0)
    sd_min = np.min(sd_data, axis=0)
    # differential data
    tmp_diff = np.diff(sd_data, axis=0)
    tmp_diff = tmp_diff / sd_data[1:, :]
    sd_max_diff = np.max(tmp_diff, axis=0)
    sd_min_diff = np.min(tmp_diff, axis=0)

    # windowing
    sd_ma_data_5 = rolling_apply(fun_mean, sd_data, 5)  # 5days moving average
    sd_ma_data_10 = rolling_apply(fun_mean, sd_data, 10)
    sd_ma_data_20 = rolling_apply(fun_mean, sd_data, 20)
    sd_ma_data_60 = rolling_apply(fun_mean, sd_data, 60)
    fund_his_30 = rolling_apply(fun_cumsum, returns, 30)  # 30days cumulative sum

    # set cut-off 
    sd_dates_train, sd_dates_test = cut_off_data(dates, cut_off, blind_set_seq)
    sd_data_train, sd_data_test = cut_off_data(sd_data, cut_off, blind_set_seq)
    sd_ma_data_5_train, sd_ma_data_5_test = cut_off_data(sd_ma_data_5, cut_off, blind_set_seq)
    sd_ma_data_10_train, sd_ma_data_10_test = cut_off_data(sd_ma_data_10, cut_off, blind_set_seq)
    sd_ma_data_20_train, sd_ma_data_20_test = cut_off_data(sd_ma_data_20, cut_off, blind_set_seq)
    sd_ma_data_60_train, sd_ma_data_60_test = cut_off_data(sd_ma_data_60, cut_off, blind_set_seq)
    fund_his_30_train, fund_his_30_test = cut_off_data(fund_his_30, cut_off, blind_set_seq)
    if dependent_var == 'returns':
        target_data_train, target_data_test = cut_off_data(returns, cut_off, blind_set_seq)
    elif dependent_var == 'log_returns':
        target_data_train, target_data_test = cut_off_data(log_returns, cut_off, blind_set_seq)
    elif dependent_var == 'tri':
        target_data_train, target_data_test = cut_off_data(fund_data, cut_off, blind_set_seq)

    """Write examples
    """
    # generate the training and validation sets.
    convert_dataset(sd_dates_train, sd_data_train, sd_ma_data_5_train, sd_ma_data_10_train, sd_ma_data_20_train,
                    sd_ma_data_60_train, target_data_train, fund_his_30_train,
                    x_seq, forward_ndx, class_names_to_ids, dataset_dir, verbose=0)

    # blind set
    convert_dataset(sd_dates_test, sd_data_test, sd_ma_data_5_test, sd_ma_data_10_test, sd_ma_data_20_test,
                    sd_ma_data_60_test, target_data_test, fund_his_30_test,
                    x_seq, forward_ndx, class_names_to_ids, dataset_dir, verbose=1)

    # write meta information for data set
    meta = {'x_seq': x_seq, 'x_variables': x_variables, 'forecast': forward_ndx,
            'num_fund': num_fund, 'num_of_datatype_obs': 5,
            'action_to_fund': ids_to_class_names, 'fund_to_action': class_names_to_ids}
    with open(dataset_dir + './meta', 'wb') as fp:
        pickle.dump(meta, fp)
        fp.close()

    print('\nFinished converting the dataset!')
    print('\n Location: {0}'.format(dataset_dir))
