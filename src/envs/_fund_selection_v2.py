import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import util
import tensorflow as tf
from datasets.protobuf2tensor import DataSet
from datasets.fs_by_tf_v0 import features_description
import warnings
import sys
class FundSelectionEnv(gym.Env):

    def __init__(self):

        self.num_fund = 349
        self.num_index = 60
        self.window_size = 20
        self.num_of_datatype_obs = 5  # e.g. diff, diff_ma5, ... , diff_ma60
        self.current_step = 0
        self.current_episode = 0
        # Todo: cheek it later
        self.h_factor = 1E-12  # history score

        self.so = None
        self.episode = None
        self.sample = None

        # Todo: cheek it later
        global data_low, data_high
        data_low = -0.35
        data_high = 0.2

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
        # y_return_seq = self.sample['structure/class/seq_ratio']

        """fund index extraction with corresponding actions
        """
        # extract funds corresponding actions
        selected_action = np.squeeze(np.argwhere(action == 1))
        # Todo: check it later, in case of MultiDiscrete action, simply ignore CASH action
        selected_action = np.delete(selected_action, np.argwhere(selected_action == 0))

        num_selected_action = len(selected_action)
        # print('cnt of selected action: {}'.format(num_selected_action))

        """ extract data for reward calculation with action 
        """
        # Todo: adopt allocation rate later...
        # extract data from given actions
        fund_cum_return = fund_cum_return[:, selected_action]
        fund_min_return = fund_min_return[selected_action]
        fund_max_return = fund_max_return[selected_action]
        y_return = y_return[selected_action]
        y_return_ref0 = y_return_ref0[selected_action]
        y_return_ref1 = y_return_ref1[selected_action]
        y_return_ref2 = y_return_ref2[selected_action]
        # y_return_seq = y_return_seq[selected_action]

        """reward calculation
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # history based score (fund cumulative returns)
                history_score = (fund_cum_return[-1] - fund_min_return + 1E-5) / (fund_max_return - fund_min_return)
                history_score = 1 / np.where(history_score == 0, 1, history_score)
                history_score = history_score * self.h_factor

                # expectation of fund tri return
                expectation = (y_return_ref0 * 0.75) + (y_return * 1) + (y_return_ref1 * 0.5) + (y_return_ref2 * 0.25)
                expectation = expectation * history_score
                expectation = np.sum(expectation) / num_selected_action
                reward = expectation
            except Warning:
                raise ValueError('Check reward calculation!!!')

        """evaluation of reward
        """
        condition01 = reward
        condition02 = (np.sum(y_return) / num_selected_action)
        done = condition01 < 0 \
               or condition02 < 0
        done = bool(done)
        if not done:
            # print('reward: {:.1}/{:.1}, {}/{}'.format(condition01, condition02,
            #                                           self.current_step, self.current_episode))
            # next observation
            if (self.current_step + 1) == len(self.episode):
                self.current_step = 0
                self.current_episode = self.current_episode + 1
                self.episode = self.so.extract_samples()
            else:
                self.current_step = self.current_step + 1
            self.sample = self.episode[self.current_step]
            self.state = self._get_observation_from_sample(self.sample)
        elif self.steps_beyond_done is None:  # == (self.current_step = 0)
            # call reset function
            self.steps_beyond_done = 0
            # print('reward: {:.1}/{:.1}, {}/{}'.format(condition01, condition02,
            #                                           self.current_step, self.current_episode))
        else:  # never happen
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- "
                    "any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            # print('drop reward: {:.1}/{:.1}, {}/{}'.format(condition01, condition02,
            #                                                self.current_step, self.current_episode))
            reward = 0.0

        sys.stdout.write('\r>> [Action: %d] Reward:  %f/%f, %d/4-%d' % (num_selected_action,
                                                                        condition01*1E10, condition02*1E2,
                                                                        self.current_step, self.current_episode))
        sys.stdout.flush()

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.current_episode = self.current_episode + 1
        self.steps_beyond_done = None

        self.episode = self.so.extract_samples()
        self.sample = self.episode[self.current_step]
        self.state = self._get_observation_from_sample(self.sample)

        return np.array(self.state)

    def render(self, mode='human'):
        if self.state is None: return None

        # # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def method_test(self, var):
        return print('method_test: {}'.format(var))

    def _get_observation_from_sample(self, sample):
        # sample shape: [window size X num of indices]
        return np.concatenate((sample['structure/diff'], sample['structure/diff_ma5'],
                               sample['structure/diff_ma10'], sample['structure/diff_ma20'],
                               sample['structure/diff_ma60']), axis=0). \
            reshape((self.num_of_datatype_obs, self.window_size, self.num_index))
