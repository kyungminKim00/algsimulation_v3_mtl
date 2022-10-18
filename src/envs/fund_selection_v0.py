import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import util
import warnings
import sys
import pickle
import header.fund_selection.RUNHEADER as RUNHEADER
from functools import lru_cache

class FundSelectionEnv(gym.Env):

    def __init__(self, write_file=False):

        self.write_file = write_file
        if self.write_file:
            self.fp_expectation = open('./expectation.txt', 'a')
            self.fp_y_return = open('./y_return.txt', 'a')
            self.fp_y_return_ref1 = open('./y_return_ref1.txt', 'a')
            self.fp_y_return_ref2 = open('./y_return_ref2.txt', 'a')
            self.fp_history_score = open('./history_score.txt', 'a')
            self.fp_penalty_lookup = open('./penalty_lookup.txt', 'a')

        with open(RUNHEADER.m_dataset_dir+'/meta', 'rb') as fp:
            meta = pickle.load(fp)
            fp.close()

        self.num_fund = meta['num_fund']
        self.num_index = meta['x_variables']
        self.window_size = meta['x_seq']
        self.num_of_datatype_obs = meta['num_of_datatype_obs']  # e.g. diff, diff_ma5, ... , diff_ma60
        self.action_to_fund = meta['action_to_fund']
        self.fund_to_action = meta['fund_to_action']

        self.so = None
        self.current_episode_idx = None
        self.current_step = None
        self.episode = None
        self.sample = None
        self.mode = None
        self.eoe = False
        self.eof = False
        self.progress_info = False
        self.reward_list = list()
        self.h_factor = RUNHEADER.m_h_factor
        self.factor = RUNHEADER.m_factor

        # Todo: cheek it later
        global data_low, data_high
        # diff
        # data_low = -0.35
        # data_high = 0.2

        # normal
        data_low = -np.inf
        data_high = np.inf

        # self.action_space = spaces.Discrete(self.num_fund)  # number os funds
        # self.action_space = spaces.Box(low=0, high=1, shape=(self.num_fund))
        self.action_space = spaces.MultiDiscrete(np.ones(self.num_fund) * 2, )  # number os funds - [noop / buy]
        self.observation_space = spaces.Box(low=data_low, high=data_high,
                                            shape=(self.num_of_datatype_obs, self.window_size, self.num_index),
                                            dtype=np.float32)

        self.seed()
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Todo: calculate reward with current step action and obs(fund_history data)
        """sample data binding
        """
        # a cumulative tri return with  30days moving Avg. [window size, number of funds]
        fund_cum_return = self.sample['fund_his/data_cumsum30']
        # min tri returns of 'fund_his/data_cumsum30' data [number of funds]
        fund_min_return = self.sample['fund_his/patch_min']
        # max tri returns of 'fund_his/data_cumsum30' data [number of funds]
        fund_max_return = self.sample['fund_his/patch_max']
        # expectation of fund returns
        y_return = self.sample['structure/class/ratio']  # +5
        y_return_ref0 = self.sample['structure/class/ratio_ref0']  # +1 return
        y_return_ref1 = self.sample['structure/class/ratio_ref1']  # +10 return
        y_return_ref2 = self.sample['structure/class/ratio_ref2']  # +20 return
        y_return_ref3 = self.sample['structure/class/ratio_ref3']  # +0 return
        # y_return_ref4 = self.sample['structure/class/ratio_ref4']  # +30 return
        # y_return_ref5 = self.sample['structure/class/ratio_ref5']  # +40 return
        # y_return_ref6 = self.sample['structure/class/ratio_ref6']  # +50 return
        # y_return_ref7 = self.sample['structure/class/ratio_ref7']  # +60 return
        # y_return_seq = self.sample['structure/class/seq_ratio']

        """fund index extraction with corresponding actions
        """
        # extract funds corresponding actions
        selected_action = np.squeeze(np.argwhere(action == 1))
        # Todo: check it later, in case of MultiDiscrete action, simply ignore CASH action
        selected_action = np.delete(selected_action, np.argwhere(selected_action == 0))

        num_selected_action = len(selected_action)
        # print('cnt of selected action: {}'.format(num_selected_action))

        # Todo: dummy action, Warning!!!
        if num_selected_action == 0:
            # print('\ndummy action fired!!!!!!\n')
            selected_action = np.squeeze(np.argwhere(action == 0))
            selected_action = np.delete(selected_action, np.argwhere(selected_action == 0))
            num_selected_action = len(selected_action)

        """ extract data for reward calculation with action 
        """
        # Todo: adopt allocation rate later...
        # extract data from given actions
        fund_cum_return = fund_cum_return[selected_action] / num_selected_action
        fund_min_return = fund_min_return[selected_action] / num_selected_action
        fund_max_return = fund_max_return[selected_action] / num_selected_action
        y_return = y_return[selected_action] / num_selected_action
        y_return_ref0 = y_return_ref0[selected_action] / num_selected_action
        y_return_ref1 = y_return_ref1[selected_action] / num_selected_action
        y_return_ref2 = y_return_ref2[selected_action] / num_selected_action
        y_return_ref3 = y_return_ref3[selected_action] / num_selected_action
        # y_return_ref4 = y_return_ref4[selected_action] / num_selected_action
        # y_return_ref5 = y_return_ref5[selected_action] / num_selected_action
        # y_return_ref6 = y_return_ref6[selected_action] / num_selected_action
        # y_return_ref7 = y_return_ref7[selected_action] / num_selected_action
        # y_return_seq = y_return_seq[selected_action]

        """ Caution: not in use for training,  
            used in performance calculation and tensor-board only
        """
        info = {
            'selected_fund_name': [self.action_to_fund[idx] for idx in selected_action],
            'selected_action': selected_action.tolist(),
            'date': self.sample['date/base_date_label'],
            '0day_return': np.sum(y_return_ref3),  # current returns
            '1day_return': np.sum(y_return_ref0),  # current +1 day returns
            '5day_return': np.sum(y_return),  # current +5 returns
            '10day_return': np.sum(y_return_ref1),  # current +10 returns
            '20day_return': np.sum(y_return_ref2),  # current +20 day returns
            # '30day_return': np.sum(y_return_ref4),  # current +30 day returns
            # '40day_return': np.sum(y_return_ref5),  # current +40 day returns
            # '50day_return': np.sum(y_return_ref6),  # current +50 day returns
            # '60day_return': np.sum(y_return_ref7),  # current +60 day returns
        }

        """ Reward calculation
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # history of the fund penalty (using fund cumulative returns)
                history_score = (fund_cum_return - fund_min_return + 1E-5) / (fund_max_return - fund_min_return)
                history_score = np.mean(history_score) * self.h_factor

                # num of action penalty
                total = self.num_fund
                target = RUNHEADER.m_target_actions
                penalty_lookup = np.array((np.linspace(1, 0, target).tolist() +
                                           np.linspace(0, 1, target)[1:].tolist() +
                                           [1] * (total + 1 - (target * 2)))) * self.factor
                # penalty_lookup = np.array([0]*349)

                # expectation with tri return
                # expectation = (y_return_ref0 * 0.25) + (y_return * 1.5) + (y_return_ref1 * 0.5) + (y_return_ref2 * 0.25)
                expectation = (y_return * 1) + (y_return_ref1 * 0.35) + (y_return_ref2 * 0.15)

                """
                Expectation
                """
                expectation = np.sum(expectation)
                done_cond = np.sum(y_return)
                done_cond2 = np.sum(y_return_ref1)
                done_cond3 = np.sum(y_return) > np.sum(y_return_ref1 - y_return)

                # for co-relation coefficient analysis
                if self.write_file:
                    print(expectation, file=self.fp_expectation)
                    print(done_cond, file=self.fp_y_return)
                    print(np.sum(y_return_ref1), file=self.fp_y_return_ref1)
                    print(np.sum(y_return_ref2), file=self.fp_y_return_ref2)
                    print(history_score/self.h_factor, file=self.fp_history_score)
                    print(penalty_lookup[num_selected_action]/self.factor, file=self.fp_penalty_lookup)

                expectation = expectation - penalty_lookup[num_selected_action] + history_score
                reward = expectation

                # # reward = expectation
                # if expectation > 0.8:
                #     reward = expectation
                # elif (expectation <= 0.8) and (expectation >= -0.7):
                #     reward = 0
                # else:
                #     reward = expectation

                # reward = 1
            except Warning:
                pass
                # raise ValueError('Check reward calculation!!!')

        """evaluation of reward
        """
        try:
            """
            (num_selected_action < RUNHEADER.m_allow_actions_min) or : All the step sample should satisfy this condition
            (num_selected_action > RUNHEADER.m_allow_actions_max) or : All the step sample should satisfy this condition
            (expectation < 0) or : All the step sample should satisfy this condition
            (done_cond < 0) or (done_cond2 < 0) or : Give advantage to the sample when meet this condition
            
            # Give advantage to the sample when meet this condition
            (np.sum(y_return) > np.sum(y_return_ref1 - y_return)) : cool-down phase  
            """
            option_cond = [done_cond < 0, done_cond2 < 0, done_cond3]  # done_cond3: cool-down phase
            option_cond = [True for item in option_cond if not item]  # count the number of False
            if (np.sum(np.array(option_cond))) >= 2:
                option_cond = False
            else:
                option_cond = True

            done = (num_selected_action < RUNHEADER.m_allow_actions_min) or \
                   (num_selected_action > RUNHEADER.m_allow_actions_max) or \
                   (expectation < 0) or option_cond
            done = bool(done)
        except ValueError:
            print('here here')

        if self.mode == 'train':
            self.state = self.next_timestamp(self.current_episode_idx, self.current_step)
            # self.state, done = self.train_step(done)
        elif self.mode == 'test':
            self.state, done = self.test_step(self.current_step)

        return np.array(self.state), reward, done, info

    def test_step(self, current_step=0):
        self.eof = False
        # self.current_step = self.current_step + 1
        if self.current_step < len(self.episode):
            self.sample = self.episode[current_step]
            self.state = self._get_observation_from_sample(self.sample)
        else:
            self.eof = True
        return self.state, self.eof

    def reset(self):
        if self.steps_beyond_done == 0:  # done==True
            self.steps_beyond_done = None
            raise ValueError('Disable for now')
            return np.array(self.next_timestamp(self.current_episode_idx, self.current_step))
        else:  # when init stage
            self.steps_beyond_done = 0
            return np.array(self.next_timestamp(self.current_episode_idx, self.current_step, init=True))

    def render(self, mode='human'):
        if self.state is None: return None
        return None

    def close(self):
        return None

    def get_progress_info(self):
        return self.progress_info

    def _get_observation_from_sample(self, sample):
        # sample shape: [window size X num of indices]
        return np.concatenate((sample['structure/normal'], sample['structure/normal_ma5'],
                               sample['structure/normal_ma10'], sample['structure/normal_ma20'],
                               sample['structure/normal_ma60']), axis=0). \
            reshape((self.num_of_datatype_obs, self.window_size, self.num_index))

    @lru_cache(maxsize=RUNHEADER.m_n_step)
    def next_timestamp(self, current_episode_idx=0, current_step=0, init=False):
        if init:
            self.steps_beyond_done = None
        self.episode, self.progress_info, self.eoe = self.so.extract_samples(current_episode_idx)

        try:
            self.sample = self.episode[current_step]
        except IndexError:  # at the end point of the episode
            self.episode, self.progress_info, self.eoe = self.so.extract_samples(current_episode_idx)
            # self.current_step = 0
            self.sample = self.episode[current_step]

        # print(self.sample['date/base_date_label'])
        return self._get_observation_from_sample(self.sample)

    def clear_cache(self):
        self.next_timestamp.cache_clear()