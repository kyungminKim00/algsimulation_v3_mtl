import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd


class FundSelectionEnv(gym.Env):

    def __init__(self):

        self.num_fund = 349
        self.num_index = 62
        self.window_size = 20
        self.ahead = 5

        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5 # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates
        # self.kinematics_integrator = 'euler'
        #
        # # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4

        self.tmp_delete = 0
        global data_low, data_high

        data = np.random.rand(self.window_size, self.num_index, 10)  # observation days:, variables, samples
        # data_low = np.min(np.min(data, axis=0), axis=0)
        # data_high = np.max(np.max(data, axis=0), axis=0)

        data_low = -0.5
        data_high = 0.5

        self.action_space = spaces.Discrete(self.num_fund)  # number os funds
        self.observation_space = spaces.Box(low=data_low, high=data_high,
                                            shape=(self.window_size, self.num_index, 3),
                                            dtype=np.float32)  # observation days, 1, number of variables

        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state

        # x, x_dot, theta, theta_dot = state
        tmp_observation = state
        # print(tmp_observation)

        # force = self.force_mag if action == 1 else -self.force_mag
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #             self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # if self.kinematics_integrator == 'euler':
        #     x = x + self.tau * x_dot
        #     x_dot = x_dot + self.tau * xacc
        #     theta = theta + self.tau * theta_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        # else:  # semi-implicit euler
        #     x_dot = x_dot + self.tau * xacc
        #     x = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot

        # self.state = (x, x_dot, theta, theta_dot)
        self.state = self.np_random.uniform(low=-0.5, high=0.5,
                               size=(self.window_size, self.num_index, 3))

        # delete
        self.tmp_delete = self.tmp_delete + 1
        self.state = self.state + self.tmp_delete

        # done = x < -self.x_threshold \
        #        or x > self.x_threshold \
        #        or theta < -self.theta_threshold_radians \
        #        or theta > self.theta_threshold_radians
        done = self.np_random.uniform(low=-0.5, high=0.5) < 0  # constraint satisfaction (delete)
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # Feed Random index TFrecord data
        self.state = self.np_random.uniform(low=-0.5, high=0.5,
                                            size=(self.window_size, self.num_index, 3))
        # test state delete
        self.tmp_delete = self.tmp_delete + 1
        self.state = np.zeros(shape=(self.window_size, self.num_index, 3)) + self.tmp_delete

        self.steps_beyond_done = None
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
