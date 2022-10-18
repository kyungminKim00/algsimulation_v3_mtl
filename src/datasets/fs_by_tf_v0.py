from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import pickle

with open('./save/tf_record/fund_selection/meta', 'rb') as fp:
    meta = pickle.load(fp)
    fp.close()

x_seq = meta['x_seq']
num_structure_variables = meta['x_variables']
forecast = meta['forecast']
num_fund = meta['num_fund']
features_description = {
    'structure/data': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/data_max': tf.io.FixedLenFeature([num_structure_variables], tf.float32),
    'structure/data_min': tf.io.FixedLenFeature([num_structure_variables], tf.float32),
    'structure/data_ma5': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/data_ma10': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/data_ma20': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/data_ma60': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/normal': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/normal_ma5': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/normal_ma10': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/normal_ma20': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/normal_ma60': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/diff': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/diff_ma5': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/diff_ma10': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/diff_ma20': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/diff_ma60': tf.io.FixedLenFeature([x_seq, num_structure_variables], tf.float32),
    'structure/height': tf.io.FixedLenFeature([1], tf.int64),
    'structure/width': tf.io.FixedLenFeature([1], tf.int64),
    # 'structure/class/seq_price': tf.VarLenFeature([], tf.float32),
    'structure/class/seq_ratio': tf.io.FixedLenFeature([forecast, num_fund], tf.float32),
    'structure/class/seq_height': tf.io.FixedLenFeature([1], tf.int64),
    'structure/class/seq_width': tf.io.FixedLenFeature([1], tf.int64),
    # 'structure/class/label': int64_feature(patch_sd.class_id),
    # 'structure/class/price': float_feature(patch_sd.class_price),
    'structure/class/ratio': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref0': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref1': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref2': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref3': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref4': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref5': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref6': tf.io.FixedLenFeature([num_fund], tf.float32),
    'structure/class/ratio_ref7': tf.io.FixedLenFeature([num_fund], tf.float32),
    'fund_his/cov60': tf.io.FixedLenFeature([num_fund, num_fund], tf.float32),
    'fund_his/data_cumsum30': tf.io.FixedLenFeature([x_seq, num_fund], tf.float32),
    'fund_his/height': tf.io.FixedLenFeature([1], tf.int64),
    'fund_his/width': tf.io.FixedLenFeature([1], tf.int64),
    'fund_his/patch_min': tf.io.FixedLenFeature([num_fund], tf.float32),
    'fund_his/patch_max': tf.io.FixedLenFeature([num_fund], tf.float32),
    'date/base_date_label': tf.io.FixedLenFeature([1], tf.string),
    'date/base_date_index': tf.io.FixedLenFeature([1], tf.int64),
    'date/prediction_date_label': tf.io.FixedLenFeature([1], tf.string),
    'date/prediction_date_index': tf.io.FixedLenFeature([1], tf.int64),
}

# Example
if __name__ == '__main__':  # For unit test

    # method 1. outer session
    from datasets.protobuf2tensor import DataSet

    with tf.Graph().as_default():
        # [Default] Test configuration
        # split_name='test', n_batch_size=1, n_shuffle_buffer_size=1, n_epoch=1
        # shuffle_method='batch_shuffle', sliding_batch=False, s_window=1, s_stride=1,
        dataset = DataSet(dataset_dir='../save/tf_record/fund_selection', file_pattern='fs_v2_cv%02d_%s.tfrecord',
                          split_name='test', cv_number=0, n_batch_size=1, n_shuffle_buffer_size=10, n_epoch=2,
                          shuffle_method='batch_shuffle', sliding_batch=True, s_window=5, s_stride=1,
                          features_description=features_description)
        k = 0
        with tf.compat.v1.Session() as sess:
            for i in range(dataset.n_epoch):  # iterate with n_epoch
                dataset.re_init()
                while not dataset.eof:  # get whole samples in a batch
                    try:
                        if not dataset.eof:  # 1 epoch
                            tmp = dataset.extract_samples(sess)
                            print(tmp['date/base_date_label'])
                            k = k + 1
                            print(str(k))

                    except UnboundLocalError:
                        sys.stdout.write('\r>> Feed samples for {} epoch'.format(i))
                        sys.stdout.flush()

