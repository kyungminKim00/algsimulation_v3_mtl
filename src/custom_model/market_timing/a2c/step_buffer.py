import numpy as np


class StepBuffer(object):
    def __init__(self, env):
        self.n_env = env.num_envs

        if len(env.observation_space.shape) > 1:
            self.raw_pixels = True
            self.height, self.width, self.n_channels = env.observation_space.shape
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
        self.dones = None
        self.returns_info = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def put(self, actions, values, states, enc_obs, rewards, dones,
            returns_returns_info, mode=False, min_reward=0, disable_dones=False):

        _dones = np.reshape(dones, [self.n_env])
        cond_idx = np.squeeze(np.argwhere(_dones == 0), axis=1).tolist()

        env_idx = list()
        for idx in cond_idx:
            env_idx.append(idx)

        try:
            if len(env_idx) > 0:
                if self.enc_obs is None:
                    self.enc_obs = np.empty([self.n_env] + list(enc_obs.shape[1:]), dtype=self.obs_dtype)
                    self.actions = np.empty([self.n_env] + list(actions.shape[1:]), dtype=np.int32)
                    self.rewards = np.empty([self.n_env], dtype=np.float32)
                    self.values = np.empty([self.n_env], dtype=np.float32)
                    self.states = np.empty([self.n_env] + list(states.shape[1:]), dtype=np.float32)
                    self.dones = np.empty([self.n_env], dtype=np.bool)
                    self.returns_returns_info = np.empty([self.n_env]).tolist()

                for idx in env_idx:
                    self.enc_obs[self.next_idx] = np.reshape(enc_obs,
                                                             [self.n_env] + list(enc_obs.shape[1:]))[idx]
                    self.actions[self.next_idx] = np.reshape(actions,
                                                             [self.n_env] + list(actions.shape[1:]))[idx]
                    self.rewards[self.next_idx] = np.reshape(rewards,
                                                             [self.n_env])[idx]
                    self.values[self.next_idx] = np.reshape(values,
                                                            [self.n_env])[idx]
                    self.dones[self.next_idx] = np.reshape(dones,
                                                           [self.n_env])[idx]
                    self.states[self.next_idx] = states[idx]

                    self.returns_returns_info[self.next_idx] = returns_returns_info[idx]

                    self.next_idx = (self.next_idx + 1) % self.n_env
                    self.num_in_buffer = min(self.n_env, self.num_in_buffer + 1)

        except TypeError:
            print("Type Error")

    def take(self, arr, idx, envx, dummy=False):
        n_env = len(envx)
        if dummy:
            out = list()
            for i in range(n_env):
                out.append(arr[idx[i]])
        else:
            out = np.empty([n_env] + list(arr.shape[1:]), dtype=arr.dtype)
            for i in range(n_env):
                out[i] = arr[idx[i]]
        return out

    # episode experience pop
    def get(self):
        n_env = self.n_env
        idx = np.random.randint(0, self.num_in_buffer, n_env)

        if self.num_in_buffer == n_env:
            idx = np.arange(n_env)
        envx = np.arange(n_env)

        returns_returns_info = self.take(self.returns_returns_info, idx, envx, dummy=True)
        states = self.take(self.states, idx, envx)
        actions = self.take(self.actions, idx, envx)
        rewards = self.take(self.rewards, idx, envx)
        values = self.take(self.values, idx, envx)
        dones = self.take(self.dones, idx, envx)
        obs = self.take(self.enc_obs, idx, envx)

        # actions, values, states, obs, rewards, dones, returns_returns_info = \
        #     self.reshape(obs, actions, rewards, values, states, dones, returns_returns_info)

        return actions, values, states, obs, rewards, dones, returns_returns_info

    def reshape(self, obs, actions, rewards, values, states, dones, returns_returns_info):
        obs = np.reshape(obs, [self.n_env] + list(obs.shape[2:]))
        actions = np.reshape(actions, [self.n_env] + list(actions.shape[2:]))
        rewards = np.reshape(rewards, [self.n_env])
        values = np.reshape(values, [self.n_env])
        states = np.reshape(states, states.shape)
        dones = np.reshape(dones, [self.n_env])

        return actions, values, states, obs, rewards, dones, returns_returns_info
