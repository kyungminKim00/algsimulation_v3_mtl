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
import os
import random
import sys

import tensorflow as tf
import numpy as np
from datasets.windowing import rolling_apply, fun_mean
from datasets import dataset_utils
from datasets.dataset_utils import float_feature, int64_feature, bytes_feature
import pandas as pd
import datetime

# The URL where the Flowers data can be downloaded.
# _DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.
# _NUM_VALIDATION = 350

# Seed for repeatability.
# _RANDOM_SEED = 0


class ReadData(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(
        self, date, data, target_data, x_seq, forward_ndx, class_names_to_ids, data_type
    ):
        self.source_data = data
        self.target_data = target_data
        self.x_seq = x_seq
        self.forward_ndx = forward_ndx
        self.class_names_to_ids = class_names_to_ids
        self.data_type = data_type
        self.date = date

    def _get_patch(self, base_date):
        x_start_ndx = base_date - self.x_seq + 1
        x_end_ndx = base_date + 1
        forward_ndx = self.forward_ndx

        """X data Section
        """
        if self.data_type == 0:  # structure data
            # price data
            self.data = self.source_data[
                x_start_ndx:x_end_ndx, :
            ]  # x_seq+1 by the num of varianbles
            self.height, self.width = self.data.shape

            # normalize data (volatility)
            self.normal_data = (self.data - np.mean(self.data, axis=0)) / np.std(
                self.data, axis=0
            )

            assert (self.data.shape[0] != self.normal_data.shape[0]) or (
                self.data.shape[1] != self.normal_data.shape[1]
            )

            # differential data
            self.diff_data = np.diff(
                self.source_data[x_start_ndx - 1 : x_end_ndx, :], axis=0
            )
            self.diff_data = (
                self.diff_data / self.source_data[x_start_ndx - 1 : x_end_ndx - 1, :]
            )
        else:  # un-structure data
            None

        """Y data Section
        """
        self.class_seq_price = self.target_data[
            base_date + 1 : base_date + forward_ndx + 1
        ]  # y_seq by the num of variables
        self.class_seq_ratio = (
            self.class_seq_price
            / self.target_data[base_date - forward_ndx + 1 : base_date + 1]
        ) - 1
        self.class_price = self.target_data[base_date + forward_ndx]
        self.class_ratio = (self.class_price / self.target_data[base_date]) - 1

        if self.class_ratio >= 0:
            self.class_name = "up"
        else:
            self.class_name = "down"
        self.class_id = self.class_names_to_ids[self.class_name]

        """Date data Section
        """
        self.base_date_index = base_date
        self.base_date_label = self.date[base_date]
        self.prediction_date_index = base_date + forward_ndx
        self.prediction_date_label = self.date[base_date + forward_ndx]

    def get_patch(self, base_date):
        self.data = None
        self.height = None
        self.width = None
        self.normal_data = None
        self.diff_data = None

        self.class_seq_price = None
        self.class_seq_ratio = None
        self.class_price = None
        self.class_ratio = None
        self.class_id = None
        self.class_name = None

        self.base_date_label = None
        self.base_date_index = None
        self.prediction_date_label = None
        self.prediction_date_index = None

        # extract a patch
        self._get_patch(base_date)


# def _get_filenames_and_classes(dataset_dir):
#  """Returns a list of filenames and inferred class names.
#
#  Args:
#    dataset_dir: A directory containing a set of subdirectories representing
#      class names. Each subdirectory should contain PNG or JPG encoded images.
#
#  Returns:
#    A list of image file paths, relative to `dataset_dir` and the list of
#    subdirectories, representing class names.
#  """
#  flower_root = os.path.join(dataset_dir, 'flower_photos')
#  directories = []
#  class_names = []
#  for filename in os.listdir(flower_root):
#    path = os.path.join(flower_root, filename)
#    if os.path.isdir(path):
#      directories.append(path)
#      class_names.append(filename)
#
#  photo_filenames = []
#  for directory in directories:
#    for filename in os.listdir(directory):
#      path = os.path.join(directory, filename)
#      photo_filenames.append(path)
#
#  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, split_id, shard_id):
    output_filename = "snp_v1_%s_cv%03d_%03d-of-%05d.tfrecord" % (
        split_name,
        split_id,
        shard_id,
        _NUM_SHARDS,
    )
    return "{0}/{1}".format(dataset_dir, output_filename)


def convert_dataset(
    sd_dates,
    sd_data,
    sd_ma_data_5,
    sd_ma_data_10,
    sd_ma_data_20,
    sd_ma_data_60,
    ud_data,
    target_data,
    x_seq,
    forward_ndx,
    class_names_to_ids,
    dataset_dir,
    verbose,
):
    """Converts the given filenames to a TFRecord dataset.

    Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
    (integers).
    dataset_dir: The directory where the converted datasets are stored.
    """

    num_per_shard = int(math.ceil(len(sd_dates) / float(_NUM_SHARDS)))
    date = sd_dates

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
    ud_reader = ReadData(
        date, ud_data, target_data, x_seq, forward_ndx, class_names_to_ids
    )

    # with tf.Session('') as sess:
    #     if verbose == 0:  # train and validation
    #         for split_id in range(_NUM_SHARDS):
    #             for shard_id in range(_NUM_SHARDS):
    #                 _convert_dataset(date, sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20,
    #                                  sd_reader_ma60, ud_reader, x_seq, forward_ndx, dataset_dir,
    #                                  num_per_shard, shard_id, split_id, verbose)
    #
    #     else:  # blind set (test)
    #         _convert_dataset(date, sd_reader, sd_reader_ma5, sd_reader_ma10, sd_reader_ma20,
    #                          sd_reader_ma60, ud_reader, x_seq, forward_ndx, dataset_dir,
    #                          num_per_shard, 1, 1, verbose)

    if verbose == 0:  # train and validation
        for split_id in range(_NUM_SHARDS):
            for shard_id in range(_NUM_SHARDS):
                _convert_dataset(
                    date,
                    sd_reader,
                    sd_reader_ma5,
                    sd_reader_ma10,
                    sd_reader_ma20,
                    sd_reader_ma60,
                    ud_reader,
                    x_seq,
                    forward_ndx,
                    dataset_dir,
                    num_per_shard,
                    shard_id,
                    split_id,
                    verbose,
                )

    else:  # blind set (test)
        _convert_dataset(
            date,
            sd_reader,
            sd_reader_ma5,
            sd_reader_ma10,
            sd_reader_ma20,
            sd_reader_ma60,
            ud_reader,
            x_seq,
            forward_ndx,
            dataset_dir,
            num_per_shard,
            1,
            1,
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
    ud_reader,
    x_seq,
    forward_ndx,
    dataset_dir,
    num_per_shard,
    shard_id,
    split_id,
    verbose,
):
    if verbose == 0:
        if split_id == shard_id:
            output_filename = _get_dataset_filename(
                dataset_dir, "validation", split_id, shard_id
            )
        else:
            output_filename = _get_dataset_filename(
                dataset_dir, "train", split_id, shard_id
            )
    else:
        output_filename = _get_dataset_filename(dataset_dir, "test", 1, 1)

    with tf.Graph().as_default():
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(date))
        for i in range(start_ndx, end_ndx):
            sys.stdout.write(
                "\r>> Converting data %s shard %d of cv %d"
                % (str(date[i]), shard_id, split_id)
            )
            sys.stdout.flush()

            # Read Data
            if ((i - x_seq) > 0) and ((i + forward_ndx) < end_ndx):
                sd_reader.get_patch(i)
                sd_reader_ma5.get_patch(i)
                sd_reader_ma10.get_patch(i)
                sd_reader_ma20.get_patch(i)
                sd_reader_ma60.get_patch(i)
                ud_reader.get_patch(i)

                example = _snp_v1_to_tfexample(
                    sd_reader,
                    sd_reader_ma5,
                    sd_reader_ma10,
                    sd_reader_ma20,
                    sd_reader_ma60,
                    ud_reader,
                )
                tfrecord_writer.write(example.SerializeToString())


# def _clean_up_temporary_files(dataset_dir):
#  """Removes temporary files used to create the dataset.
#
#  Args:
#    dataset_dir: The directory where the temporary files are stored.
#  """
#  filename = _DATA_URL.split('/')[-1]
#  filepath = os.path.join(dataset_dir, filename)
#  tf.gfile.Remove(filepath)
#
#  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
    for split_name in ["train", "validation"]:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
            if not tf.io.gfile.exists(output_filename):
                return False
    return True


def _snp_v1_to_tfexample(
    patch_sd, patch_sd_ma5, patch_sd_ma10, patch_sd_ma20, patch_sd_ma60, patch_ud
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
                "structure/diff": float_feature(patch_sd.diff_data),
                "structure/diff_ma5": float_feature(patch_sd_ma5.diff_data),
                "structure/diff_ma10": float_feature(patch_sd_ma10.diff_data),
                "structure/diff_ma20": float_feature(patch_sd_ma20.diff_data),
                "structure/diff_ma60": float_feature(patch_sd_ma60.diff_data),
                "structure/class/label": int64_feature(patch_sd.class_id),
                "structure/class/seq_price": float_feature(patch_sd.class_seq_price),
                "structure/class/seq_ratio": float_feature(patch_sd.class_seq_ratio),
                "structure/class/price": float_feature(patch_sd.class_price),
                "structure/class/ratio": float_feature(patch_sd.class_ratio),
                "structure/height": int64_feature(patch_sd.height),
                "structure/width": int64_feature(patch_sd.width),
                "unstructure/data": float_feature(patch_ud.data),
                "unstructure/height": int64_feature(patch_ud.height),
                "unstructure/width": int64_feature(patch_ud.width),
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


def has_negative_values(data, keys, target_str):
    target_index = keys.get_loc(target_str)

    # find negative-valued index
    del_idx = np.argwhere(np.sum(np.where(data < 0, True, False), axis=0) > 0)
    if len(del_idx) > 0:
        del_idx = del_idx.reshape(len(del_idx))

    # add target index
    del_idx = np.append(del_idx, target_index)
    del_idx = np.unique(del_idx)

    all_idx = np.arange(len(keys))
    positive_value_idx = np.delete(all_idx, del_idx)

    assert len(positive_value_idx) == 0

    data = data[positive_value_idx]
    keys = keys[positive_value_idx]

    return data, keys


def check_nan(data, keys):
    check = np.argwhere(np.sum(np.isnan(data), axis=0) == 1)
    if len(check) > 0:
        ValueError("{0} contains nan values".format(keys[check.reshape(len(check))]))


def get_operation_dates_data(dates, data, target_str, pair_str, keys):
    if keys.contains(target_str) and keys.contains(pair_str):
        target_index = keys.get_loc(target_str)
        pair_index = keys.get_loc(pair_str)
    else:
        raise ValueError("raw data does not contains target or pair index !!!")

    checksum_data = data.T[[target_index, pair_index]].T
    non_operational_dates_index = np.argwhere(
        np.sum(np.diff(checksum_data.T).T, axis=1) == 0
    )
    non_operational_dates_index = non_operational_dates_index + 1

    cnt = len(non_operational_dates_index)
    if cnt > 0:
        non_operational_dates_index = list(non_operational_dates_index.reshape(cnt))

        # get operational dates index
        tmp_idx = [
            idx for idx in range(len(dates)) if idx not in non_operational_dates_index
        ]
        dates = dates[tmp_idx]
        data = data[tmp_idx]
        target_data = data.T[target_index].T

    return dates, data, target_data


def get_conjunction_dates_data(sd_dates, ud_dates, sd_data, ud_data, target_data):
    sd_dates_true = np.empty(0, dtype=np.int)
    ud_dates_true = np.empty(0, dtype=np.int)
    ud_dates_true_label = np.empty(0, dtype=np.object)

    for i in range(len(sd_dates)):
        for k in range(len(ud_dates)):
            if sd_dates[i] == ud_dates[k]:  # conjunction of sd_dates and ud_dates
                if (
                    np.sum(np.isnan(ud_data[ud_dates[k]])) == 0
                ):  # ud_date should embedding vector
                    sd_dates_true = np.append(sd_dates_true, i)
                    ud_dates_true = np.append(ud_dates_true, k)
                    ud_dates_true_label = np.append(ud_dates_true_label, ud_dates[k])

    sd_dates = sd_dates[sd_dates_true]
    sd_data = sd_data[sd_dates_true]
    target_data = target_data[sd_dates_true]

    ud_dates = ud_dates[ud_dates_true]

    keys = np.array([*ud_data])
    for key in keys:
        if key not in ud_dates_true_label:
            del ud_data[key]

    assert len(sd_dates) == len(ud_dates)
    assert len(sd_dates) == len(ud_data)

    return sd_dates, sd_data, ud_data, target_data


def get_read_data(target_str, pair_str, sd_path, ud_path):
    """Validate data and Return actual operation days for target_index

    1. check nan on data
    2. check operation dates of target index
        (find working days for target index except holidays)
    3. check negative values on data

    Structure Data:
        numpy - dates by the numbers of variables
    Unstructure Data:
        dictionary - {key: the label of dates,
                      values: embedding vector (1 by 300)

    """
    sd_dates, sd_data, keys = _get_read_data(sd_path)
    ud_dates, ud_data, keys = _get_read_data(ud_path)

    # 1. nan check
    check_nan(sd_data, keys)
    check_nan(ud_data, keys)

    # 2-1. [row-wised filter] picks reference indexes to find operation dates for the target index (=e.g. snp500)
    sd_dates, sd_data, target_data = get_operation_dates_data(
        sd_dates, sd_data, target_str, pair_str, keys
    )

    # 2-2. [row-wised filter] the conjunction of structure data dates and un-structure data dates
    dates, sd_data, ud_data, target_data = get_conjunction_dates_data(
        sd_dates, ud_dates, sd_data, ud_data, target_data
    )

    # 3. [column-wised filter] negative index check (capture positive data only) and make non-ar data
    sd_data, keys = has_negative_values(sd_data, keys, target_str)

    return dates, sd_data, ud_data, keys, target_data


def _get_read_data(path):
    """Retrieve working days
    Args:
    path : raw data path

    """
    df = pd.read_csv(path)

    keys = df.columns[1:]
    data = np.array(df.values[:, 1:], dtype=np.float32)
    dates = df.values[:, 0]

    # the data from monday to friday
    working_days_index = list()
    for i in range(len(dates)):
        tmp_date = datetime.datetime.strptime(dates[i], "%Y.%m.%d")
        if tmp_date.weekday() < 5:  # keep working days
            working_days_index.append(i)
        dates[i] = tmp_date.strftime("%Y-%m-%d")

    dates = dates[working_days_index]  # re-store working days
    data = data[working_days_index]  # re-store working days

    return dates, data, keys


def cut_off_data(data, cut_off, blind_set_seq):
    eof = len(data)
    blind_set_seq = eof - blind_set_seq

    if len(data.shape) == 1:  # 1D
        tmp = data[cut_off:blind_set_seq], data[blind_set_seq:]
    elif len(data.shape) == 2:  # 2D:
        tmp = data[cut_off:blind_set_seq, :], data[blind_set_seq:, :]
    else:
        raise IndexError("Define your cut-off code")
    return tmp


def run(dataset_dir):
    """Conversion operation.
    Args:
    dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.io.gfile.exists(dataset_dir):
        tf.io.gfile.makedirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print("Dataset files already exist. Exiting without re-creating them.")
        return

    x_seq = 20  # 20days
    forward_ndx = 10  # the days after 10 days
    class_names = ["up", "down"]
    target_str = "spx index"
    pair_str = "USGG10YR Index"
    sd_raw_data = "./datasets/rawdata/sd_68.csv"
    ud_raw_data = ""
    blind_set_seq = 200
    cut_off = 70
    global sd_max, sd_min, _NUM_SHARDS

    # The number of shards per dataset split.
    _NUM_SHARDS = 5

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    """Generate re-fined data from raw data
    """
    # refined data from raw data
    sd_dates, sd_data, ud_data, sd_key, target_data = get_read_data(
        target_str, pair_str, sd_raw_data, ud_raw_data
    )

    # calculate statistics for re-fined data
    sd_max = np.max(sd_data, axis=0)
    sd_min = np.min(sd_data, axis=0)

    sd_ma_data_5 = rolling_apply(fun_mean, sd_data, 5)  # 5days moving average
    sd_ma_data_10 = rolling_apply(fun_mean, sd_data, 10)
    sd_ma_data_20 = rolling_apply(fun_mean, sd_data, 20)
    sd_ma_data_60 = rolling_apply(fun_mean, sd_data, 60)

    # set cut-off
    sd_dates_train, sd_dates_test = cut_off_data(sd_dates, cut_off, blind_set_seq)
    sd_data_train, sd_data_test = cut_off_data(sd_data, cut_off, blind_set_seq)
    target_data_train, target_data_test = cut_off_data(
        target_data, cut_off, blind_set_seq
    )
    sd_ma_data_5_train, sd_ma_data_5_test = cut_off_data(
        sd_ma_data_5, cut_off, blind_set_seq
    )
    sd_ma_data_10_train, sd_ma_data_10_test = cut_off_data(
        sd_ma_data_10, cut_off, blind_set_seq
    )
    sd_ma_data_20_train, sd_ma_data_20_test = cut_off_data(
        sd_ma_data_20, cut_off, blind_set_seq
    )
    sd_ma_data_60_train, sd_ma_data_60_test = cut_off_data(
        sd_ma_data_60, cut_off, blind_set_seq
    )

    """Write examples
    """
    # generate the training and validation sets.
    convert_dataset(
        sd_dates_train,
        sd_data_train,
        sd_ma_data_5_train,
        sd_ma_data_10_train,
        sd_ma_data_20_train,
        sd_ma_data_60_train,
        ud_data,
        target_data_train,
        x_seq,
        forward_ndx,
        class_names_to_ids,
        dataset_dir,
        verbose=0,
    )

    # blind set
    convert_dataset(
        sd_dates_test,
        sd_data_test,
        sd_ma_data_5_test,
        sd_ma_data_10_test,
        sd_ma_data_20_test,
        sd_ma_data_60_test,
        ud_data,
        target_data_test,
        x_seq,
        forward_ndx,
        class_names_to_ids,
        dataset_dir,
        verbose=1,
    )

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print("\nFinished converting the dataset!")
