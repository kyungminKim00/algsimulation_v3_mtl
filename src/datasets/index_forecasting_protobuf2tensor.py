from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datasets import dataset_utils

# from tensorflow.contrib.data.python.ops import sliding


class DataSet:
    def __init__(
        self,
        dataset_dir="../save/tf_record/fund_selection",
        file_pattern="fs_v2_cv%02d_%s.tfrecord",
        split_name="test",
        cv_number=0,
        n_batch_size=1,
        n_shuffle_buffer_size=1,
        n_epoch=1,
        shuffle_method="batch_shuffle",
        sliding_batch=False,
        s_window=5,
        s_stride=1,
        features_description=None,
    ):

        if split_name not in ["train", "validation", "test"]:
            raise ValueError("split_name is one of train, validation, test")

        file_pattern = os.path.join(dataset_dir, file_pattern % (cv_number, split_name))

        self.eof = False
        self.n_shuffle_buffer_size = n_shuffle_buffer_size
        self.n_batch_size = n_batch_size
        self.n_epoch = n_epoch
        self.features_description = features_description
        self.dataset = tf.data.TFRecordDataset(filenames=file_pattern)
        self.sliding_batch = sliding_batch
        # dataset generation
        if self.sliding_batch:
            self.parsed_dataset = self.dataset.map(self._parse_function).apply(
                sliding.sliding_window_batch(s_window, s_stride)
            )
        else:
            self.parsed_dataset = self.dataset.map(self._parse_function)
        self.examples = self._get_dataset(method=shuffle_method)
        self.keys = self.examples.keys()
        self.values = self.examples.values()
        self.shape = self.parsed_dataset

        if dataset_utils.has_labels(dataset_dir):
            self.labels_to_names = dataset_utils.read_label_file(dataset_dir)

    def _get_dataset(self, method="batch_shuffle"):
        if method == "batch_shuffle":  # apply batch size first and then shuffle
            dataset = tf.compat.v1.data.make_one_shot_iterator(
                self.parsed_dataset.batch(
                    batch_size=self.n_batch_size, drop_remainder=True
                ).shuffle(
                    buffer_size=self.n_shuffle_buffer_size,
                    reshuffle_each_iteration=True,
                )
            ).get_next()
        elif method == "sample_shuffle":  # apply shuffle first and then batch size
            dataset = tf.compat.v1.data.make_one_shot_iterator(
                self.parsed_dataset.shuffle(
                    buffer_size=self.n_shuffle_buffer_size,
                    reshuffle_each_iteration=True,
                ).batch(batch_size=self.n_batch_size, drop_remainder=True)
            ).get_next()

        return dataset

    def re_init(self):
        self.examples = self._get_dataset()
        self.eof = False

    def extract_samples(self, tf_sess):
        try:
            examples = tf_sess.run(self.examples)
        except tf.errors.OutOfRangeError:
            self.eof = True

        return examples

    def _parse_function(self, example_proto):
        return tf.io.parse_single_example(
            serialized=example_proto, features=self.features_description
        )
