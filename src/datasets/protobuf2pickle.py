from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets import dataset_utils
from datasets.windowing import rolling_window

import pickle
import numpy as np
import RUNHEADER

class DataSet:
    def __init__(self, dataset_dir='../save/tf_record/fund_selection', file_pattern='fs_v0_cv%02d_%s.pkl',
                 split_name='test', cv_number=0, n_batch_size=1):

        if split_name not in ['train', 'validation', 'test']:
            raise ValueError('split_name is one of train, validation, test')

        file_pattern = os.path.join(dataset_dir, file_pattern % (cv_number, split_name))

        # get number of agents
        m_name = RUNHEADER.m_name
        _model_location = './save/model/rllearn/' + m_name
        with open(_model_location + '/meta', mode='rb') as fp:
            meta = pickle.load(fp)
            fp.close()
        self.n_cpu = meta['_n_cpu']
        # self._n_step = meta['_n_step']

        with open(file_pattern, 'rb') as fp:
            dataset = pickle.load(fp)
            fp.close()

        self.split_name = split_name
        self.sample_idx = 0
        self.n_epoch = 0
        self.epoch_progress_train = None
        self.epoch_done = False

        if split_name == 'train':  # adopt rolling window
            self.dataset = rolling_window(np.array(dataset), n_batch_size)
            self.n_episode = int(len(self.dataset))
            self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))
        else:
            self.dataset = dataset
            self.n_episode = int(len(self.dataset))

        if dataset_utils.has_labels(dataset_dir):
            self.labels_to_names = dataset_utils.read_label_file(dataset_dir)

    def generate_shuffle_idx(self):
        self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))

    def extract_samples(self, sample_idx=0):
        if self.split_name == 'train':
            # self.distribute_same_example = self.distribute_same_example + 1
            try:
                eoe = False
                self.epoch_done = False
                examples = self.dataset[self.shuffled_episode[sample_idx]]
                # if (self.distribute_same_example % self.n_cpu) == 0:
                #     self.sample_idx = self.sample_idx + 1

                # # same example feeding for the test delete later
                # self.sample_idx = 0
            except IndexError:
                # print('extract_samples: indexError')
                eoe = True
                self.epoch_done = True
                self.n_epoch = self.n_epoch + 1

                # self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))

                # self.sample_idx = 0

                examples = self.dataset[self.shuffled_episode[0]]
                # self.sample_idx = self.sample_idx + 1
                # # re-shuffle (all samples are iterated)
                # if self.sample_idx > (self.n_episode - 1):
                #     self.epoch_done = True
                #     self.n_epoch = self.n_epoch + 1
                #     self.sample_idx = 0
                #     self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))
            # if (self.distribute_same_example % self.n_cpu) == 0:
            #     self.sample_idx = self.sample_idx + 1
            #     print('self.distribute_same_example: {}, self.sample_idx: {}'.format(self.distribute_same_example, self.sample_idx))
        else:  # test
            eoe = False
            self.epoch_done = True
            self.sample_idx = self.n_episode
            self.n_epoch = 1

            examples = self.dataset

        # just information
        self.epoch_progress_train = {
            'contain_dates': [item['date/base_date_label'] for item in examples],
            'current_sample': sample_idx,
            'total_episode': self.n_episode,
            'n_epoch': (self.n_epoch / self.n_cpu),
            'epoch_done': self.epoch_done,
        }

        return examples, self.epoch_progress_train, eoe

    # # original
    # def extract_samples(self):
    #     if self.split_name == 'train':
    #         self.distribute_same_example = self.distribute_same_example + 1
    #         try:
    #             self.epoch_done = False
    #             examples = self.dataset[self.shuffled_episode[self.sample_idx]]
    #             # if (self.distribute_same_example % self.n_cpu) == 0:
    #             #     self.sample_idx = self.sample_idx + 1
    #
    #             # # same example feeding for the test delete later
    #             # self.sample_idx = 0
    #         except IndexError:
    #             print('extract_samples: indexError')
    #             self.epoch_done = True
    #             self.n_epoch = self.n_epoch + 1
    #             self.sample_idx = 0
    #             self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))
    #             examples = self.dataset[self.shuffled_episode[self.sample_idx]]
    #             # self.sample_idx = self.sample_idx + 1
    #             # # re-shuffle (all samples are iterated)
    #             # if self.sample_idx > (self.n_episode - 1):
    #             #     self.epoch_done = True
    #             #     self.n_epoch = self.n_epoch + 1
    #             #     self.sample_idx = 0
    #             #     self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))
    #         if (self.distribute_same_example % self.n_cpu) == 0:
    #             self.sample_idx = self.sample_idx + 1
    #             print('self.distribute_same_example: {}, self.sample_idx: {}'.format(self.distribute_same_example, self.sample_idx))
    #     else:  # test
    #         self.epoch_done = True
    #         self.sample_idx = self.n_episode
    #         self.n_epoch = 1
    #
    #         examples = self.dataset
    #
    #     # just information
    #     self.epoch_progress_train = {
    #         'contain_dates': [item['date/base_date_label'] for item in examples],
    #         'current_sample': self.sample_idx,
    #         'total_episode': self.n_episode,
    #         'n_epoch': self.n_epoch,
    #         'epoch_done': self.epoch_done,
    #     }
    #
    #     return examples, self.epoch_progress_train
