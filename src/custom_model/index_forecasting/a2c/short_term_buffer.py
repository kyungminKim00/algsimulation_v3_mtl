import numpy as np


class SBuffer(object):
    def __init__(self, env, n_steps):
        """
        A buffer for observations, actions, rewards, mu's, states, masks and dones values

        :param env: (Gym environment) The environment to learn from
        :param n_steps: (int) The number of steps to run for each environment
        :param size: (int) The buffer size in number of steps
        """
        self.n_env = env.num_envs
        self.n_steps = n_steps
        # self.n_batch = self.n_env * self.n_steps
        # # Each loc contains n_env * n_steps frames, thus total buffer is n_env * size frames
        # self.size = size // self.n_steps

        if len(env.observation_space.shape) > 1:
            self.raw_pixels = True
            (
                self.height,
                self.width,
                self.n_channels,
                self.n_targets,
            ) = env.observation_space.shape
            self.obs_dtype = np.float32
        else:
            self.raw_pixels = False
            if len(env.observation_space.shape) == 1:
                self.obs_dim = env.observation_space.shape[-1]
            else:
                self.obs_dim = 1
            self.obs_dtype = np.float32

        # Memory
        self.enc_obs = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.states = None
        self.masks = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def put(
        self,
        enc_obs,
        actions,
        rewards,
        values,
        states,
        masks,
        env_idx,
    ):
        """
        Adds a frame to the buffer

        :param env_idx:
        :param buffer_th:
        :param hit: ([float])
        :param returns_info: ([float])
        :param states: ([float])
        :param values: ([float])
        :param enc_obs: ([float]) the encoded observation
        :param actions: ([int]) the actions
        :param rewards: ([float]) the rewards
        :param masks: ([bool])
        """
        # enc_obs [n_env, (n_steps + n_stack), nh, nw, nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]

        # cond = np.reshape(rewards, [self.n_env, self.n_steps])
        # cond_1 = np.squeeze(np.argwhere(np.sum(cond, axis=1) > entry_th), axis=1).tolist()
        # cond = np.reshape(masks, [self.n_env, self.n_steps])
        # cond_2 = np.squeeze(np.argwhere(np.array([util.consecutive_true(item) for item in cond]) > mask_th),
        #                     axis=1).tolist()
        # env_idx = np.array(list(set(cond_1 + cond_2)))

        try:
            if len(env_idx) > 0:
                if self.enc_obs is None:
                    self.enc_obs = np.empty(
                        [self.n_env] + [self.n_steps] + list(enc_obs.shape[1:]),
                        dtype=self.obs_dtype,
                    )
                    self.actions = np.empty(
                        [self.n_env] + [self.n_steps] + list(actions.shape[1:]),
                        dtype=np.int32,
                    )
                    self.rewards = np.empty(
                        [self.n_env] + [self.n_steps] + list(rewards.shape[1:]),
                        dtype=np.float32,
                    )
                    self.values = np.empty(
                        [self.n_env, self.n_steps] + list(values.shape[1:]),
                        dtype=np.float32,
                    )
                    self.states = np.empty(
                        [self.n_env] + list(states.shape[1:]), dtype=np.float32
                    )
                    self.masks = np.empty(
                        [self.n_env, self.n_steps] + list(masks.shape[1:]),
                        dtype=np.bool,
                    )

                for idx in env_idx:
                    self.enc_obs[self.next_idx] = np.reshape(
                        enc_obs, [self.n_env, self.n_steps] + list(enc_obs.shape[1:])
                    )[idx]
                    self.actions[self.next_idx] = np.reshape(
                        actions, [self.n_env, self.n_steps] + list(actions.shape[1:])
                    )[idx]
                    self.rewards[self.next_idx] = np.reshape(
                        rewards, [self.n_env, self.n_steps] + list(rewards.shape[1:])
                    )[idx]
                    self.values[self.next_idx] = np.reshape(
                        values, [self.n_env, self.n_steps] + list(values.shape[1:])
                    )[idx]
                    self.masks[self.next_idx] = np.reshape(
                        masks, [self.n_env, self.n_steps] + list(masks.shape[1:])
                    )[idx]
                    self.states[self.next_idx] = states[idx]

                    self.next_idx = (self.next_idx + 1) % self.n_env
                    self.num_in_buffer = min(self.n_env, self.num_in_buffer + 1)

        except TypeError:
            print("Type Error")

    def take(self, arr, idx, envx, dummy=False):
        """
        Reads a frame from a list and index for the asked environment ids

        :param arr: (np.ndarray) the array that is read
        :param idx: ([int]) the idx that are read
        :param envx: ([int]) the idx for the environments
        :return: ([float]) the askes frames from the list
        """
        n_env = len(envx)
        if dummy:  # not in use for replay experience
            # out = np.empty(list(arr.shape[1:]), dtype=arr.dtype)
            # out = arr[idx[0]]
            out = np.empty([n_env] + list(arr.shape[1:]), dtype=arr.dtype)
            for i in range(n_env):
                out[i] = arr[idx[i]]
        else:
            out = np.empty([n_env] + list(arr.shape[1:]), dtype=arr.dtype)
            for i in range(n_env):
                out[i] = arr[idx[i]]
        return out

    # episode experience pop
    def get(self):
        """
        randomly read a frame from the buffer

        :return: ([float], [float], [float], [float], [bool], [float])
                 observations, actions, rewards, mus, dones, maskes
        """
        # returns
        # obs [n_env, (n_steps + 1), nh, nw, n_stack*nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]
        n_env = self.n_env

        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, n_env)
        envx = np.arange(n_env)

        states = self.take(self.states, idx, envx)
        actions = self.take(self.actions, idx, envx)
        rewards = self.take(self.rewards, idx, envx)
        values = self.take(self.values, idx, envx)
        masks = self.take(self.masks, idx, envx)
        obs = self.take(self.enc_obs, idx, envx)
        # obs = self.decode(enc_obs)

        (obs, actions, rewards, values, states, masks,) = self.reshape(
            obs,
            actions,
            rewards,
            values,
            states,
            masks,
        )

        return (
            obs,
            actions,
            rewards,
            values,
            states,
            masks,
        )

    def reshape(
        self,
        obs,
        actions,
        rewards,
        values,
        states,
        masks,
    ):
        obs = np.reshape(obs, [self.n_env * self.n_steps] + list(obs.shape[2:]))
        actions = np.reshape(
            actions, [self.n_env * self.n_steps] + list(actions.shape[2:])
        )
        rewards = np.reshape(
            rewards, [self.n_env * self.n_steps] + list(rewards.shape[2:])
        )
        values = np.reshape(
            values, [self.n_env * self.n_steps] + list(values.shape[2:])
        )
        states = np.reshape(states, states.shape)
        masks = np.reshape(masks, [self.n_env * self.n_steps])

        return (
            obs,
            actions,
            rewards,
            values,
            states,
            masks,
        )
