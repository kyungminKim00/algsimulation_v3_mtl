import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import warnings
import pickle
import header.market_timing.RUNHEADER as RUNHEADER
from functools import lru_cache


class MarketTimingEnv(gym.Env):

    def __init__(self, write_file=False):

        self.write_file = write_file
        if self.write_file:
            self.fp_expectation = open('./expectation.txt', 'a')
            self.fp_y_return = open('./y_return.txt', 'a')
            self.fp_y_return_ref1 = open('./y_return_ref1.txt', 'a')
            self.fp_y_return_ref2 = open('./y_return_ref2.txt', 'a')
            self.fp_history_score = open('./history_score.txt', 'a')
            self.fp_penalty_lookup = open('./penalty_lookup.txt', 'a')
            self.fp_cov = open('./cov.txt', 'a')

        with open(RUNHEADER.m_dataset_dir + '/meta', 'rb') as fp:
            meta = pickle.load(fp)
            fp.close()

        self.num_y_index = meta['num_y_index']
        self.num_index = meta['x_variables']
        self.window_size = meta['x_seq']
        self.num_of_datatype_obs = meta['num_of_datatype_obs']  # e.g. diff, diff_ma5, ... , diff_ma60
        self.num_of_datatype_obs_total = meta['num_of_datatype_obs_total']
        self.num_of_datatype_obs_total_mt = meta['num_of_datatype_obs_total_mt']
        self.action_to_y_index = meta['action_to_y_index']
        self.y_index_to_action = meta['y_index_to_action']

        self.so = None
        self.so_validation = None
        self.n_episode = None
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
        self.cov_factor = RUNHEADER.m_cov_factor

        # static information
        self.m_total_example = None
        self.timestep = None
        self.m_total_timesteps = None
        self.m_buffer_size = None
        self.m_main_replay_start = None

        # Todo: cheek it later
        global data_low, data_high
        # diff
        # data_low = -0.35
        # data_high = 0.2

        # normal
        data_low = -np.inf
        data_high = np.inf

        # self.action_space = spaces.Discrete(self.num_y_index)  # number os funds
        # self.action_space = spaces.Box(low=0, high=1, shape=(self.num_y_index))

        # up/down +20, +10, +15, +25, +30
        self.action_space = spaces.MultiDiscrete(np.ones(5) * 2, )
        self.observation_space = spaces.Box(low=data_low, high=data_high,
                                            shape=(self.num_of_datatype_obs_total, self.window_size, self.num_index),
                                            dtype=np.float32)
        self.action_space_mt = spaces.MultiDiscrete(np.ones(1) * 3, )  # B,H,S (L, H, S)
        self.observation_space_mt = spaces.Box(low=data_low, high=data_high,
                                            shape=(self.num_of_datatype_obs_total_mt, self.window_size, 1),
                                            dtype=np.float32)
        self.seed()
        self.state = None
        self.state_mt = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = None
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.mode == 'train':
            self.next_timestamp(self.current_episode_idx, self.current_step)
        else:
            self.test_step(self.current_step)

        """fund index extraction with corresponding actions
        """
        # extract y corresponding actions
        selected_action = np.squeeze(np.argwhere(action == 1))

        """ extract data for reward calculation with action 
        """
        # Todo: adopt allocation rate later...
        # extract data from given actions
        NoneType_Check = None
        target_index = RUNHEADER.m_target_index
        y_return_ratio = self.sample['structure/class/ratio'][target_index]  # ratio +20
        y_return_ratio_ref1 = self.sample['structure/class/ratio_ref1'][target_index]  # ratio +10
        y_return_index = self.sample['structure/class/index'][target_index]  # index +20

        y_return_label = self.sample['structure/class/label'][target_index]  # up/down +20
        y_return_label_ref1 = self.sample['structure/class/label_ref1'][target_index]  # up/down +10
        y_return_label_ref2 = self.sample['structure/class/label_ref2'][target_index]  # up/down +15

        y_tr_label_call = self.sample['structure/class/tr_label_call'][target_index]
        y_tr_label_hold = self.sample['structure/class/tr_label_hold'][target_index]
        y_tr_label_put = self.sample['structure/class/tr_label_put'][target_index]
        y_tr_index = self.sample['structure/class/tr_index'][target_index]

        if type(NoneType_Check) is type(self.sample['structure/class/label_ref4']):  # NoneType is not subscriptable
            # fix it later
            dummy = y_return_label_ref2
            y_return_label_ref4 = dummy
            y_return_label_ref5 = dummy
        else:
            y_return_label_ref4 = self.sample['structure/class/label_ref4'][target_index]  # up/down +25
            y_return_label_ref5 = self.sample['structure/class/label_ref5'][target_index]  # up/down +30

        y_return = np.array([y_return_label, y_return_label_ref1, y_return_label_ref2,
                             y_return_label_ref4, y_return_label_ref5], dtype=np.int)
        y_return_seq_ratio = self.sample['structure/class/seq_ratio'][:, target_index]  # -2 ~ +2 (5days)
        y_current_index = self.sample['structure/class/base_date_price'][target_index]  # +0 index

        """ Caution: not in use for training,  
            used in performance calculation and tensor-board only
        """
        info = {
            'selected_action_label': ['P_20days', 'P_10days', 'P_15days', 'P_25days', 'P_30days'],
            'selected_action': action.tolist(),
            'real_action': y_return.tolist(),
            'date': self.sample['date/base_date_label'],
            'p_date': self.sample['date/prediction_date_label'],
            'today_index': y_current_index,  # +0 index
            '20day_index': y_return_index,  # +20 index
            '20day_return': y_return_ratio,  # +20 returns
            '20day_label': y_return_label,  # +20 label
            '10day_return': y_return_ratio_ref1,  # +10 returns
        }

        """ Reward calculation
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:

                """
                Expectation
                """
                if RUNHEADER.m_n_cpu == 1:  # use mean
                    expectation = np.mean(y_return_seq_ratio)
                else:
                    expectation = y_return_seq_ratio[np.random.randint(0, y_return_seq_ratio.shape[0])]
                done_cond = np.sum(np.abs(action - y_return)) > 0
                done_cond2 = np.abs(action - y_return)[0] > 0 or np.sum(np.abs(action - y_return)) > 1
                expectation_mt = y_tr_index

                reward = expectation
                reward_mt = expectation_mt
            except Warning:
                pass
                # raise ValueError('Check reward calculation!!!')

        """evaluation of reward
        """
        try:
            done = done_cond2
            done = bool(done)
        except ValueError:
            print('here here')

        return np.array(self.state), reward, done, info, np.array(self.state_mt), reward_mt

    @lru_cache(maxsize=RUNHEADER.blind_set_seq)
    def test_step(self, current_step=0):
        self.eof = False

        if self.current_step < self.n_episode:
            if self.mode == 'validation':
                self.sample, _, _ = self.so.extract_samples(0, current_step)
            else:  # test
                self.sample, _, _ = self.so.extract_samples(0, current_step)
            self.state, self.state_mt = self._get_observation_from_sample(self.sample)
        else:
            self.eof = True

    def get_total_episode(self):
        return self.so.get_total_episode()

    def get_timestep(self):
        return self.so.get_timestep()

    def get_total_timesteps(self):
        return self.so.get_total_timesteps()

    def get_buffer_size(self):
        return self.so.get_buffer_size()

    def get_main_replay_start(self):
        return self.so.get_main_replay_start()

    def reset(self):
        if self.steps_beyond_done == 0:  # done==True
            self.steps_beyond_done = None
            raise ValueError('Disable for now')
            return np.array(self.next_timestamp(self.current_episode_idx, self.current_step)[0])
        else:  # when init stage
            self.steps_beyond_done = 0
            if self.mode == 'train':
                self.n_episode = self.so.get_total_episode()
            else:  # validation and test
                if self.mode == 'validation':
                    # # validation but memory leak, not fixed yet disable this code
                    # self.n_episode = self.so_validation.get_total_episode()

                    # alternative method inference separately - just run twice as test mode
                    self.n_episode = self.so.get_total_episode()
                else:  # test
                    self.n_episode = self.so.get_total_episode()
            self.next_timestamp(self.current_episode_idx, self.current_step, init=True)

            return np.array(self.state), np.array(self.state_mt)

    def render(self, mode='human'):
        if self.state is None: return None
        return None

    def close(self):
        return None

    def get_progress_info(self):
        return self.progress_info

    # @util.funTime('_get_observation_from_sample')
    def _get_observation_from_sample(self, sample):
        return sample['structure/predefined_observation_total'], sample['structure/predefined_observation_ar']

    @lru_cache(maxsize=RUNHEADER.m_n_step)
    def next_timestamp(self, current_episode_idx=0, current_step=0, init=False):
        self.sample = None

        if init:
            self.steps_beyond_done = None

        if self.mode == 'train':
            self.episode, self.progress_info, self.eoe = self.so.extract_samples(current_episode_idx)
            self.sample = self.episode[current_step]
        elif self.mode == 'validation':  # validation
            # # validation but memory leak, not fixed yet disable this code
            # self.sample, self.progress_info, self.eoe = \
            #     self.so_validation.extract_samples(current_episode_idx, current_step)

            # alternative method inference separately - just run twice as test mode
            self.sample, self.progress_info, self.eoe = self.so.extract_samples(current_episode_idx, current_step)
        else:  # test
            self.sample, self.progress_info, self.eoe = self.so.extract_samples(current_episode_idx, current_step)
        self.state, self.state_mt = self._get_observation_from_sample(self.sample)

    def obs_from_env(self):
        self.episode, self.progress_info, self.eoe = self.so.extract_samples(self.current_episode_idx)
        return self._get_observation_from_sample(self.episode[self.current_step])

    def clear_cache(self):
        self.next_timestamp.cache_clear()

    def clear_cache_test_step(self):
        self.test_step.cache_clear()
