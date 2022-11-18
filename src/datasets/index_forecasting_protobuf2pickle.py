from __future__ import absolute_import, division, print_function

import os
import pickle

import numpy as np
from header.index_forecasting import RUNHEADER
from util import funTime, json2dict

from datasets import dataset_utils, generate_val_test_with_X
from datasets.windowing import rolling_window


def get_x_dates(model_location, tf_record_location, forward_ndx):
    x_dict = json2dict("{}/selected_x_dict.json".format(model_location))
    try:
        with open("{}/meta".format(tf_record_location), "rb") as fp:
            info = pickle.load(fp)
            s_test = info["test_set_start"]
            e_test = info["test_set_end"]
            fp.close()
    except FileNotFoundError:
        s_test = None
        e_test = None

    return x_dict, s_test, e_test, forward_ndx


class DataSet:
    @funTime("Loading data")
    def __init__(
        self,
        dataset_dir="../save/tf_record/index_forecasting",
        file_pattern="if_v0_cv%02d_%s.pkl",
        split_name="test",
        cv_number=0,
        n_batch_size=1,
        regenerate=False,
        model_location=None,
        forward_ndx=None,
        patch_data=None,
    ):

        if split_name not in ["train", "validation", "test"]:
            raise ValueError("split_name is one of train, validation, test")
        # if split_name == 'validation':
        #     split_name = 'test'

        file_pattern = os.path.join(dataset_dir, file_pattern % (cv_number, split_name))

        # get number of agents
        m_name = RUNHEADER.m_name
        _model_location = "./save/model/rllearn/" + m_name

        with open(_model_location + "/meta", mode="rb") as fp:
            meta = pickle.load(fp)
            fp.close()
        self.n_cpu = meta["_n_cpu"]
        # self._n_step = meta['_n_step']

        if not regenerate:
            with open(file_pattern, "rb") as fp:
                dataset = pickle.load(fp)
                fp.close()
        else:
            dataset = patch_data

        self.split_name = split_name
        self.sample_idx = 0
        self.n_epoch = 0
        self.epoch_progress = None
        self.epoch_done = False
        self.m_buffer_size = 0
        self.m_total_example = 0
        self.timestep = 0
        self.m_total_timesteps = 0
        self.m_buffer_size = 0
        self.m_main_replay_start = 0

        if split_name == "train":  # adopt rolling window
            self.dataset = rolling_window(np.array(dataset), n_batch_size)
            if not RUNHEADER.weighted_random_sample:
                self.shuffled_episode = np.random.permutation(np.arange(self.n_episode))
                self.n_episode = int(len(self.dataset))
            else:
                # for temporal use 40*9
                tmp = None
                n_samples = int(len(self.dataset))
                aa = np.random.permutation(np.arange(n_samples))
                bb = np.random.permutation(
                    np.arange(n_samples)[
                        int(n_samples - RUNHEADER.m_augmented_sample) :
                    ]
                )
                cc = np.random.permutation(
                    np.arange(n_samples)[
                        int(n_samples - RUNHEADER.m_augmented_sample + 20) :
                    ]
                )
                for _iter in range(RUNHEADER.m_augmented_sample_iter):
                    if _iter == 0:
                        tmp = np.append(aa, bb)
                    else:
                        if _iter < 2:  # 3 times over-populate for recent 40days
                            tmp = np.append(tmp, bb)
                        else:  # 6 times over-populate for recent 20days
                            tmp = np.append(tmp, cc)
                self.shuffled_episode = np.random.permutation(tmp)
                self.n_episode = int(len(tmp))
            # store samples information
            self.m_total_example = self.n_episode + RUNHEADER.m_augmented_sample
            self.timestep = RUNHEADER.m_n_step * self.n_cpu * self.m_total_example
            self.m_total_timesteps = (
                self.timestep * 1
            )  # time steps for generate samples
            self.m_buffer_size = int(
                self.m_total_example
                * self.n_cpu
                * RUNHEADER.m_n_step
                * RUNHEADER.buffer_drop_rate
            )  # real buffer size per buffer batch file
            self.m_main_replay_start = int(self.m_total_example * 0.99)
            # write shuffled index
            with open(_model_location + "/shuffled_episode_index.txt", mode="w") as fp:
                print(self.shuffled_episode, file=fp)
                fp.close()
            with open(_model_location + "/shuffled_episode_index", mode="wb") as fp:
                pickle.dump(self.shuffled_episode, fp)
                fp.close()
        else:
            self.dataset = dataset
            self.m_total_example = self.n_episode = int(len(self.dataset))

        if dataset_utils.has_labels(dataset_dir):
            self.labels_to_names = dataset_utils.read_label_file(dataset_dir)

    # def get_total_episode(self):
    #     return self.n_episode

    def get_total_episode(self):
        return self.m_total_example

    def get_timestep(self):
        return self.timestep

    def get_total_timesteps(self):
        return self.m_total_timesteps

    def get_buffer_size(self):
        return self.m_buffer_size

    def get_main_replay_start(self):
        return self.m_main_replay_start

    def extract_samples(self, sample_idx=0, current_step=None):
        if self.split_name == "train":
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

            # just information
            self.epoch_progress = {
                "contain_dates": [item["date/base_date_label"] for item in examples],
                "current_sample": sample_idx,
                "total_episode": self.n_episode,
                "n_epoch": int(self.n_epoch / self.n_cpu),
                "epoch_done": self.epoch_done,
            }

        else:  # test
            eoe = False
            self.epoch_done = True
            self.sample_idx = self.n_episode
            self.n_epoch = 1
            examples = self.dataset[
                current_step
            ]  # train and test data structures are different (keep it mind)

            # just information
            self.epoch_progress = {
                "contain_dates": [examples["date/base_date_label"]],
                "current_sample": current_step,
                "total_episode": self.n_episode,
                "n_epoch": int(self.n_epoch / self.n_cpu),
                "epoch_done": self.epoch_done,
            }

        return examples, self.epoch_progress, eoe
