import pickle
import warnings
from functools import lru_cache

import gym
import header.index_forecasting.RUNHEADER as RUNHEADER
import numpy as np
from gym import spaces
from gym.utils import seeding


class IndexForecastingEnv(gym.Env):
    def __init__(self, write_file=False):

        self.write_file = write_file
        if self.write_file:
            self.fp_expectation = open("./expectation.txt", "a")
            self.fp_y_return = open("./y_return.txt", "a")
            self.fp_y_return_ref1 = open("./y_return_ref1.txt", "a")
            self.fp_y_return_ref2 = open("./y_return_ref2.txt", "a")
            self.fp_history_score = open("./history_score.txt", "a")
            self.fp_penalty_lookup = open("./penalty_lookup.txt", "a")
            self.fp_cov = open("./cov.txt", "a")

        with open(RUNHEADER.m_dataset_dir + "/meta", "rb") as fp:
            meta = pickle.load(fp)
            fp.close()

        self.num_y_index = meta["num_y_index"]
        self.num_index = meta["x_variables"]
        self.window_size = meta["x_seq"]
        self.num_of_datatype_obs = meta[
            "num_of_datatype_obs"
        ]  # e.g. diff, diff_ma5, ... , diff_ma60
        self.num_of_datatype_obs_total = meta["num_of_datatype_obs_total"]
        self.action_to_y_index = meta["action_to_y_index"]
        self.y_index_to_action = meta["y_index_to_action"]

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

        # cheek it later
        # global data_low, data_high

        # diff
        # data_low = -0.35
        # data_high = 0.2

        # normal
        data_low = -np.inf
        data_high = np.inf

        # self.action_space = spaces.Discrete(self.num_y_index)  # number os funds
        # self.action_space = spaces.Box(low=0, high=1, shape=(self.num_y_index))

        # up/down +20, +10, +15, +25, +30
        self.action_space = spaces.MultiDiscrete(
            np.ones(5) * 2,
        )
        self.observation_space = spaces.Box(
            low=data_low,
            high=data_high,
            shape=(self.num_of_datatype_obs_total, self.window_size, self.num_index),
            dtype=np.float32,
        )
        self.seed()
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = None
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        if self.mode == "train":
            self.next_timestamp(self.current_episode_idx, self.current_step)
        else:
            self.test_step(self.current_step)

        """fund index extraction with corresponding actions
        """
        # extract y corresponding actions
        # selected_action = np.squeeze(np.argwhere(action == 1))

        """ extract data for reward calculation with action
        """
        # adopt allocation rate later...
        # extract data from given actions
        NoneType_Check = None
        target_index = RUNHEADER.m_target_index
        y_return_ratio = self.sample["structure/class/ratio"][target_index]  # ratio +20
        y_return_ratio_ref1 = self.sample["structure/class/ratio_ref1"][
            target_index
        ]  # ratio +10
        y_return_index = self.sample["structure/class/index"][target_index]  # index +20

        y_return_label = self.sample["structure/class/label"][
            target_index
        ]  # up/down +20
        y_return_label_ref1 = self.sample["structure/class/label_ref1"][
            target_index
        ]  # up/down +10
        y_return_label_ref2 = self.sample["structure/class/label_ref2"][
            target_index
        ]  # up/down +15
        if type(NoneType_Check) is type(
            self.sample["structure/class/label_ref4"]
        ):  # NoneType is not subscriptable
            # fix it later
            dummy = y_return_label_ref2
            y_return_label_ref4 = dummy
            y_return_label_ref5 = dummy
        else:
            y_return_label_ref4 = self.sample["structure/class/label_ref4"][
                target_index
            ]  # up/down +25
            y_return_label_ref5 = self.sample["structure/class/label_ref5"][
                target_index
            ]  # up/down +30

        y_return = np.array(
            [
                y_return_label,
                y_return_label_ref1,
                y_return_label_ref2,
                y_return_label_ref4,
                y_return_label_ref5,
            ],
            dtype=np.int,
        )
        y_return_seq_ratio = self.sample["structure/class/seq_ratio"][
            :, target_index
        ]  # -2 ~ +2 (5days)
        y_current_index = self.sample["structure/class/base_date_price"][
            target_index
        ]  # +0 index

        """ Caution: not in use for training,
            used in performance calculation and tensor-board only
        """
        info = {
            "selected_action_label": [
                "P_20days",
                "P_10days",
                "P_15days",
                "P_25days",
                "P_30days",
            ],
            "selected_action": action.tolist(),
            "real_action": y_return.tolist(),
            "date": self.sample["date/base_date_label"],
            "p_date": self.sample["date/prediction_date_label"],
            "today_index": y_current_index,  # +0 index
            "20day_index": y_return_index,  # +20 index
            "20day_return": y_return_ratio,  # +20 returns
            "20day_label": y_return_label,  # +20 label
            "10day_return": y_return_ratio_ref1,  # +10 returns
        }

        """ Reward calculation
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:

                # # history of the fund penalty (using fund cumulative returns)
                # if is_zero_action:
                #     history_score = 0
                # else:
                #     history_score = (fund_cum_return - fund_min_return + 1E-5) / (fund_max_return - fund_min_return)
                # history_score = np.mean(history_score) * self.h_factor
                #
                # # num of action penalty
                # total = self.num_y_index
                # target = RUNHEADER.m_target_actions
                # penalty_lookup = np.array((np.linspace(1, 0, target).tolist() +
                #                            np.linspace(0, 1, target)[1:].tolist() +
                #                            [1] * (total + 1 - (target * 2)))) * self.factor
                # # penalty_lookup = np.array([0]*349)
                #
                # # expectation with tri return
                # # expectation = (y_return_ref0 * 0.25) + (y_return * 1.5) + (y_return_ref1 * 0.5) + (y_return_ref2 * 0.25)
                # expectation = (y_return_ref0 * 1) + (y_return * 1) + (y_return_ref1 * 0.35) + (y_return_ref2 * 0.15)

                """
                Expectation
                """
                if RUNHEADER.m_n_cpu == 1:  # use mean
                    expectation = np.mean(y_return_seq_ratio)
                else:
                    expectation = y_return_seq_ratio[
                        np.random.randint(0, y_return_seq_ratio.shape[0])
                    ]
                # done_cond = np.sum(np.abs(action - y_return)) > 0
                done_cond2 = (
                    np.abs(action - y_return)[0] > 0
                    or np.sum(np.abs(action - y_return)) > 1
                )

                # # for co-relation coefficient analysis
                # if self.write_file:
                #     print(expectation, file=self.fp_expectation)
                #     print(done_cond, file=self.fp_y_return)
                #     print(np.sum(y_return_ref1), file=self.fp_y_return_ref1)
                #     print(np.sum(y_return_ref2), file=self.fp_y_return_ref2)
                #     print(history_score, file=self.fp_history_score)
                #     print(penalty_lookup[num_selected_action], file=self.fp_penalty_lookup)
                #     print('{},{}'.format(self.sample['date/base_date_label'], fund_cov_return), file=self.fp_cov)
                #
                # expectation = expectation - (np.abs(expectation) * fund_cov_return * self.cov_factor)
                # expectation = expectation - penalty_lookup[num_selected_action] + history_score
                # if is_zero_action:
                #     expectation = 0

                reward = expectation
            except Warning:
                pass
                # raise ValueError('Check reward calculation!!!')

        """evaluation of reward
        """
        try:
            # """
            # (num_selected_action < RUNHEADER.m_allow_actions_min) or : All the step sample should satisfy this condition
            # (num_selected_action > RUNHEADER.m_allow_actions_max) or : All the step sample should satisfy this condition
            # (expectation < 0) or : All the step sample should satisfy this condition
            # (done_cond < 0) or (done_cond2 < 0) or : Give advantage to the sample when meet this condition
            #
            # # Give advantage to the sample when meet this condition
            # (np.sum(y_return) > np.sum(y_return_ref1 - y_return)) : cool-down phase
            # """
            # option_cond = [done_cond < 0, done_cond2 < 0, done_cond3]  # done_cond3: cool-down phase
            # option_cond = [True for item in option_cond if not item]  # count the number of False
            # if (np.sum(np.array(option_cond))) >= 2:
            #     option_cond = False
            # else:
            #     option_cond = True
            #
            # done = (num_selected_action < RUNHEADER.m_allow_actions_min) or \
            #        (num_selected_action > RUNHEADER.m_allow_actions_max) or \
            #        (expectation < 0) or option_cond
            done = done_cond2
            done = bool(done)
        except ValueError:
            print("here here")

        return np.array(self.state), reward, done, info

    @lru_cache(maxsize=RUNHEADER.blind_set_seq)
    def test_step(self, current_step=0):
        self.eof = False

        # if self.current_step < len(self.episode):
        #     self.sample = self.episode[current_step]
        #     self.state = self._get_observation_from_sample(self.sample)
        # else:
        #     self.eof = True
        if self.current_step < self.n_episode:
            if self.mode == "validation":
                # # validation but memory leak, not fixed yet disable this code
                # self.sample, _, _ = self.so_validation.extract_samples(0, current_step)

                # alternative method inference separately - just run twice as test mode
                self.sample, _, _ = self.so.extract_samples(0, current_step)
            else:  # test
                self.sample, _, _ = self.so.extract_samples(0, current_step)
            self.state = self._get_observation_from_sample(self.sample)
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

    def reset(self, **kwargs):
        if self.steps_beyond_done == 0:  # done==True
            self.steps_beyond_done = None
            raise ValueError("Disable for now")
            # return np.array(
            #     self.next_timestamp(self.current_episode_idx, self.current_step)[0]
            # )
        else:  # when init stage
            self.steps_beyond_done = 0
            if self.mode == "train":
                self.n_episode = self.so.get_total_episode()
            else:  # validation and test
                if self.mode == "validation":
                    # # validation but memory leak, not fixed yet disable this code
                    # self.n_episode = self.so_validation.get_total_episode()

                    # alternative method inference separately - just run twice as test mode
                    self.n_episode = self.so.get_total_episode()
                else:  # test
                    self.n_episode = self.so.get_total_episode()
            self.next_timestamp(self.current_episode_idx, self.current_step, init=True)

            return np.array(self.state)

    def render(self, mode="human"):
        if self.state is None and mode == "human":
            return None
        return None

    def close(self):
        return None

    def get_progress_info(self):
        return self.progress_info

    # @util.funTime('_get_observation_from_sample')
    def _get_observation_from_sample(self, sample):
        return sample["structure/predefined_observation_total"]

    @lru_cache(maxsize=RUNHEADER.m_n_step)
    def next_timestamp(self, current_episode_idx=0, current_step=0, init=False):
        self.sample = None

        if init:
            self.steps_beyond_done = None

        if self.mode == "train":
            self.episode, self.progress_info, self.eoe = self.so.extract_samples(
                current_episode_idx
            )
            self.sample = self.episode[current_step]
        elif self.mode == "validation":  # validation
            # # validation but memory leak, not fixed yet disable this code
            # self.sample, self.progress_info, self.eoe = \
            #     self.so_validation.extract_samples(current_episode_idx, current_step)

            # alternative method inference separately - just run twice as test mode
            self.sample, self.progress_info, self.eoe = self.so.extract_samples(
                current_episode_idx, current_step
            )
        else:  # test
            self.sample, self.progress_info, self.eoe = self.so.extract_samples(
                current_episode_idx, current_step
            )
        self.state = self._get_observation_from_sample(self.sample)

    def obs_from_env(self):
        self.episode, self.progress_info, self.eoe = self.so.extract_samples(
            self.current_episode_idx
        )
        return self._get_observation_from_sample(self.episode[self.current_step])

    def clear_cache(self):
        self.next_timestamp.cache_clear()

    def clear_cache_test_step(self):
        self.test_step.cache_clear()
