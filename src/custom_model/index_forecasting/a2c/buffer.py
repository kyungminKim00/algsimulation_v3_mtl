import numpy as np
# from memory_profiler import profile


class Buffer(object):
    def __init__(self, env, n_steps, size=50000):
        """
        A buffer for observations, actions, rewards, mu's, states, masks and dones values

        :param env: (Gym environment) The environment to learn from
        :param n_steps: (int) The number of steps to run for each environment
        :param size: (int) The buffer size in number of steps
        """
        self.n_env = env.num_envs
        self.n_steps = n_steps
        self.n_batch = self.n_env * self.n_steps
        # Each loc contains n_env * n_steps frames, thus total buffer is n_env * size frames
        self.size = size // self.n_steps

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
        self.masks = None

        self.suessor = None
        self.selected_action = None
        self.diff_selected_action = None
        self.returns_info = None
        self.hit = None
        self.bin_rewards = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    # @profile
    def append(self, obj):
        self.size = self.size + obj.size
        self.next_idx = self.next_idx + obj.next_idx
        self.num_in_buffer = self.num_in_buffer + obj.num_in_buffer

        self.enc_obs = np.concatenate((self.enc_obs, obj.enc_obs), axis=0)
        self.actions = np.concatenate((self.actions, obj.actions), axis=0)
        self.rewards = np.concatenate((self.rewards, obj.rewards), axis=0)
        self.values = np.concatenate((self.values, obj.values), axis=0)
        self.states = np.concatenate((self.states, obj.states), axis=0)
        self.masks = np.concatenate((self.masks, obj.masks), axis=0)

        # Blow are dummy variables for session run.. Not in use, even tensorboard
        self.suessor = np.concatenate((self.suessor, obj.suessor), axis=0)
        self.selected_action = np.concatenate((self.selected_action, obj.selected_action), axis=0)
        self.diff_selected_action = np.concatenate((self.diff_selected_action, obj.diff_selected_action), axis=0)
        self.returns_info = np.concatenate((self.returns_info, obj.returns_info), axis=0)
        self.hit = np.concatenate((self.hit, obj.hit), axis=0)
        self.bin_rewards = np.concatenate((self.bin_rewards, obj.bin_rewards), axis=0)

    def has_atleast(self, frames):
        """
        Check to see if the buffer has at least the asked number of frames
        
        :param frames: (int) The number of frames checked
        :return: (bool) number of frames in buffer >= number asked
        """
        # Frames per env, so total (n_env * frames) Frames needed
        # Each buffer loc has n_env * n_steps frames
        return self.num_in_buffer >= (frames // self.n_steps)

    def can_sample(self):
        """
        Check if the buffer has at least one frame
        
        :return: (bool) if the buffer has at least one frame
        """
        return self.num_in_buffer > 0

    def decode(self, enc_obs):
        """
        Get the stacked frames of an observation
        
        :param enc_obs: ([float]) the encoded observation
        :return: ([float]) the decoded observation
        """
        # enc_obs has shape [n_envs, n_steps + 1, nh, nw, nc]
        # dones has shape [n_envs, n_steps, nh, nw, nc]
        # returns stacked obs of shape [n_env, (n_steps + 1), nh, nw, nc]
        n_env, n_steps = self.n_env, self.n_steps
        if self.raw_pixels:
            obs_dim = [self.height, self.width, self.n_channels]
        else:
            obs_dim = [self.obs_dim]

        obs = np.zeros([1, n_steps + 1, n_env] + obs_dim, dtype=self.obs_dtype)
        # [n_steps + nstack, n_env, nh, nw, nc]
        x_var = np.reshape(enc_obs, [n_env, n_steps + 1] + obs_dim).swapaxes(1, 0)
        obs[-1, :] = x_var

        if self.raw_pixels:
            obs = obs.transpose((2, 1, 3, 4, 0, 5))
        else:
            obs = obs.transpose((2, 1, 3, 0))
        return np.reshape(obs, [n_env, (n_steps + 1)] + obs_dim[:-1] + [obs_dim[-1]])

    def put(self, enc_obs, actions, rewards, values, states, masks, suessor,
            selected_action, diff_selected_action, returns_info, hit):
        """
        Adds a frame to the buffer
        
        :param buffer_th:
        :param hit: ([float])
        :param returns_info: ([float])
        :param diff_selected_action: ([int])
        :param suessor: (int)
        :param states: ([float])
        :param values: ([float])
        :param selected_action:  ([int])
        :param enc_obs: ([float]) the encoded observation
        :param actions: ([int]) the actions
        :param rewards: ([float]) the rewards
        :param masks: ([bool])
        """
        # # enc_obs [n_env, (n_steps + n_stack), nh, nw, nc]
        # # actions, rewards, dones [n_env, n_steps]
        # # mus [n_env, n_steps, n_act]
        #
        # cond = np.reshape(rewards, [self.n_env, self.n_steps])
        # cond_1 = np.squeeze(np.argwhere(np.sum(cond, axis=1) > entry_th), axis=1).tolist()
        # cond = np.reshape(masks, [self.n_env, self.n_steps])
        # cond_2 = np.squeeze(np.argwhere(np.array([util.consecutive_true(item) for item in cond]) > mask_th),
        #                     axis=1).tolist()
        # env_idx = list()
        # for idx in cond_1:
        #     if idx in cond_2:
        #         env_idx.append(idx)

        # push all experiences, already validated from short-simulation procedure
        env_idx = np.arange(self.n_env).tolist()

        try:
            if len(env_idx) > 0:
                if self.enc_obs is None:
                    self.enc_obs = np.empty([self.size] + [self.n_steps] + list(enc_obs.shape[1:]), dtype=self.obs_dtype)
                    self.actions = np.empty([self.size] + [self.n_steps] + list(actions.shape[1:]), dtype=np.int8)
                    self.rewards = np.empty([self.size, self.n_steps], dtype=np.float32)
                    self.values = np.empty([self.size, self.n_steps], dtype=np.float32)
                    self.states = np.empty([self.size] + list(states.shape[1:]), dtype=np.float32)
                    self.masks = np.empty([self.size, self.n_steps], dtype=np.bool)

                    # Blow are dummy variables for session run.. Not in use, even tensorboard
                    self.suessor = np.empty([self.size], dtype=np.int8)
                    self.selected_action = np.empty([self.size, self.n_steps], dtype=np.int8)
                    self.diff_selected_action = np.empty([self.size, self.n_steps], dtype=np.int8)
                    self.returns_info = np.empty([self.size, self.n_steps], dtype=np.float32)
                    self.hit = np.empty([self.size, self.n_steps], dtype=np.float32)
                    self.bin_rewards = np.empty([self.size], dtype=np.float32)
                    # self.suessor = np.empty([self.size] + list(np.array(suessor).shape), dtype=np.int32)
                    # self.selected_action = np.empty([self.size] + list(np.array(selected_action).shape), dtype=np.int32)
                    # self.diff_selected_action = np.empty([self.size] + list(np.array(diff_selected_action).shape),
                    #                                      dtype=np.int32)
                    # self.returns_info = np.empty([self.size] + list(np.array(returns_info).shape), dtype=np.float32)
                    # self.hit = np.empty([self.size] + list(np.array(hit).shape), dtype=np.float32)

                for idx in env_idx:
                    self.enc_obs[self.next_idx] = np.reshape(enc_obs,
                                                             [self.n_env, self.n_steps] + list(enc_obs.shape[1:]))[idx]
                    self.actions[self.next_idx] = np.reshape(actions,
                                                             [self.n_env, self.n_steps] + list(actions.shape[1:]))[idx]
                    self.rewards[self.next_idx] = np.reshape(rewards,
                                                             [self.n_env, self.n_steps])[idx]
                    self.values[self.next_idx] = np.reshape(values,
                                                            [self.n_env, self.n_steps])[idx]
                    self.masks[self.next_idx] = np.reshape(masks,
                                                           [self.n_env, self.n_steps])[idx]
                    self.states[self.next_idx] = states[idx]

                    self.suessor[self.next_idx] = suessor[idx]

                    self.selected_action[self.next_idx] = np.reshape(selected_action,
                                                            [self.n_env, self.n_steps])[idx]

                    self.diff_selected_action[self.next_idx] = np.reshape(diff_selected_action,
                                                            [self.n_env, self.n_steps])[idx]

                    self.returns_info[self.next_idx] = np.reshape(returns_info,
                                                            [self.n_env, self.n_steps])[idx]

                    self.hit[self.next_idx] = np.reshape(hit,
                                                            [self.n_env, self.n_steps])[idx]
                    self.bin_rewards[self.next_idx] = np.sum(np.reshape(rewards,
                                                             [self.n_env, self.n_steps])[idx])

                    self.next_idx = (self.next_idx + 1) % self.size
                    self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
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

    # episode experience pop - random
    def get(self, b_env_by_steps=False):
        """
        randomly read a frame from the buffer
        
        :return: ([float], [float], [float], [float], [bool], [float])
                 observations, actions, rewards, mus, dones, maskes
        """
        # returns
        # obs [n_env, (n_steps + 1), nh, nw, n_stack*nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]

        assert self.can_sample()

        idx = np.random.randint(0, self.num_in_buffer, self.n_env)
        envx = np.arange(self.n_env)

        # Todo: Sampling method for major-minor class problem,
        #  bin_rewards is the sum of returns of 20days forecasting on the bin (Default: 20 observation)
        # group_1 = np.argwhere((self.bin_rewards[:self.num_in_buffer] > 2) &
        #                       (self.bin_rewards[:self.num_in_buffer] < 8)).flatten().tolist()
        # group_2 = np.argwhere((self.bin_rewards[:self.num_in_buffer] > 2) &
        #                       (self.bin_rewards[:self.num_in_buffer] < 8)).flatten().tolist()
        # envx = list(set(group_1 + group_2))

        if b_env_by_steps:  # pop n_env by n_step
            selected_action = self.take(self.selected_action, idx, envx, dummy=True)
            diff_selected_action = self.take(self.diff_selected_action, idx, envx, dummy=True)
            returns_info = self.take(self.returns_info, idx, envx, dummy=True)
            hit = self.take(self.hit, idx, envx, dummy=True)
            actions = self.take(self.actions, idx, envx)
            rewards = self.take(self.rewards, idx, envx)
            values = self.take(self.values, idx, envx)
            masks = self.take(self.masks, idx, envx)
            obs = self.take(self.enc_obs, idx, envx)
        else:  # pop and transpose from (n_env by n_step) to (n_step by n_env)
            selected_action = self.take(self.selected_action, idx, envx, dummy=True).transpose([1, 0])
            diff_selected_action = self.take(self.diff_selected_action, idx, envx, dummy=True).transpose([1, 0])
            returns_info = self.take(self.returns_info, idx, envx, dummy=True).transpose([1, 0])
            hit = self.take(self.hit, idx, envx, dummy=True).transpose([1, 0])
            actions = self.take(self.actions, idx, envx).transpose([1, 0, 2])
            rewards = self.take(self.rewards, idx, envx).transpose([1, 0])
            values = self.take(self.values, idx, envx).transpose([1, 0])
            masks = self.take(self.masks, idx, envx).transpose([1, 0])
            obs = self.take(self.enc_obs, idx, envx).transpose([1, 0, 2, 3, 4])

        # it is a summarized values, so does not need transformation
        suessor = self.take(self.suessor, idx, envx, dummy=True)
        # actually first steps are always zero, and then snew generated by model
        states = self.take(self.states, idx, envx)

        return obs, actions, rewards, values, states, masks, suessor, selected_action, \
               diff_selected_action, returns_info, hit

        # episode experience pop - sequential
    def get_non_replacement(self, wrs, b_env_by_steps=False):
        """
        randomly read a frame from the buffer

        :return: ([float], [float], [float], [float], [bool], [float])
                 observations, actions, rewards, mus, dones, maskes
        """
        envx = np.arange(self.n_env)

        # Todo: Sampling method for major-minor class problem,
        #  bin_rewards is the sum of returns of 20days forecasting on the bin (Default: 20 observation)
        # group_1 = np.argwhere((self.bin_rewards[:self.num_in_buffer] > 2) &
        #                       (self.bin_rewards[:self.num_in_buffer] < 8)).flatten().tolist()
        # group_2 = np.argwhere((self.bin_rewards[:self.num_in_buffer] > 2) &
        #                       (self.bin_rewards[:self.num_in_buffer] < 8)).flatten().tolist()
        # envx = list(set(group_1 + group_2))

        if b_env_by_steps:  # pop n_env by n_step
            selected_action = self.take(self.selected_action, wrs, envx, dummy=True)
            diff_selected_action = self.take(self.diff_selected_action, wrs, envx, dummy=True)
            returns_info = self.take(self.returns_info, wrs, envx, dummy=True)
            hit = self.take(self.hit, wrs, envx, dummy=True)
            actions = self.take(self.actions, wrs, envx)
            rewards = self.take(self.rewards, wrs, envx)
            values = self.take(self.values, wrs, envx)
            masks = self.take(self.masks, wrs, envx)
            obs = self.take(self.enc_obs, wrs, envx)
        else:  # pop and transpose from (n_env by n_step) to (n_step by n_env)
            selected_action = self.take(self.selected_action, wrs, envx, dummy=True).transpose([1, 0])
            diff_selected_action = self.take(self.diff_selected_action, wrs, envx, dummy=True).transpose([1, 0])
            returns_info = self.take(self.returns_info, wrs, envx, dummy=True).transpose([1, 0])
            hit = self.take(self.hit, wrs, envx, dummy=True).transpose([1, 0])
            actions = self.take(self.actions, wrs, envx).transpose([1, 0, 2])
            rewards = self.take(self.rewards, wrs, envx).transpose([1, 0])
            values = self.take(self.values, wrs, envx).transpose([1, 0])
            masks = self.take(self.masks, wrs, envx).transpose([1, 0])
            obs = self.take(self.enc_obs, wrs, envx).transpose([1, 0, 2, 3, 4])

        # it is a summarized values, so does not need transformation
        suessor = self.take(self.suessor, wrs, envx, dummy=True)
        # actually first steps are always zero, and then snew generated by model
        states = self.take(self.states, wrs, envx)

        return obs, actions, rewards, values, states, masks, suessor, selected_action, \
           diff_selected_action, returns_info, hit

    # individual experience pop
    def pop(self, require_more_experiences):
        idx = np.random.randint(0, self.num_in_buffer, require_more_experiences)
        envx = np.arange(require_more_experiences)

        states = self.take(self.states, idx, envx)
        actions = self.take(self.actions, idx, envx)
        rewards = self.take(self.rewards, idx, envx)
        values = self.take(self.values, idx, envx)
        masks = self.take(self.masks, idx, envx)
        obs = self.take(self.enc_obs, idx, envx)
        # obs = self.decode(enc_obs)
        selected_action = self.take(self.selected_action, idx, envx, dummy=True)
        diff_selected_action = self.take(self.diff_selected_action, idx, envx, dummy=True)
        returns_info = self.take(self.returns_info, idx, envx, dummy=True)
        hit = self.take(self.hit, idx, envx, dummy=True)
        suessor = self.take(self.suessor, idx, envx, dummy=True)

        return obs, actions, rewards, values, states, masks, selected_action, \
               diff_selected_action, returns_info, hit, suessor

    # merge experiences
    def merge(self, obs, actions, rewards, values, states, masks, selected_action, \
               diff_selected_action, returns_info, hit, suessor,
              e_obs, e_actions, e_rewards, e_values, e_states, e_masks, env_idx,
              e_selected_action, e_diff_selected_action, e_returns_info, e_hit, e_suessor):

        obs_list = list()
        actions_list = list()
        rewards_list = list()
        values_list = list()
        states_list = list()
        masks_list = list()
        selected_action_list = list()
        diff_selected_action_list = list()
        returns_info_list = list()
        hit_list = list()
        suessor_list = list()

        # merge
        for idx in env_idx:
            obs_list.append(np.reshape(obs, [self.n_env, self.n_steps] + list(obs.shape[1:]))[idx])
            actions_list.append(np.reshape(actions, [self.n_env, self.n_steps] + list(actions.shape[1:]))[idx])
            rewards_list.append(np.reshape(rewards, [self.n_env, self.n_steps])[idx])
            values_list.append(np.reshape(values, [self.n_env, self.n_steps])[idx])
            states_list.append(states[idx])
            masks_list.append(np.reshape(masks, [self.n_env, self.n_steps])[idx])
            selected_action_list.append(np.reshape(selected_action, [self.n_env, self.n_steps])[idx])
            diff_selected_action_list.append(np.reshape(diff_selected_action, [self.n_env, self.n_steps])[idx])
            returns_info_list.append(np.reshape(returns_info, [self.n_env, self.n_steps])[idx])
            hit_list.append(np.reshape(hit, [self.n_env, self.n_steps])[idx])
            suessor_list.append(np.reshape(suessor, [self.n_env])[idx])

        for idx in range(len(e_rewards)):
            obs_list.append(e_obs[idx])
            actions_list.append(e_actions[idx])
            rewards_list.append(e_rewards[idx])
            values_list.append(e_values[idx])
            states_list.append(e_states[idx])
            masks_list.append(e_masks[idx])
            selected_action_list.append(e_selected_action[idx])
            diff_selected_action_list.append(e_diff_selected_action[idx])
            returns_info_list.append(e_returns_info[idx])
            hit_list.append(e_hit[idx])
            suessor_list.append(e_suessor[idx])


        # reshape
        obs, actions, rewards, values, states, masks, selected_action, diff_selected_action, returns_info, hit, suessor = \
            self.reshape(np.array(obs_list, dtype=self.obs_dtype),
                         np.array(actions_list, dtype=np.int32),
                         np.array(rewards_list, dtype=np.float32),
                         np.array(values_list, dtype=np.float32),
                         np.array(states_list, dtype=np.float32),
                         np.array(masks_list, dtype=np.bool),
                         np.array(selected_action_list, dtype=np.int32),
                         np.array(diff_selected_action_list, dtype=np.int32),
                         np.array(returns_info_list, dtype=np.float32),
                         np.array(hit_list, dtype=np.float32),
                         np.array(suessor_list, dtype=np.int32))

        return obs, actions, rewards, values, states, masks, selected_action, \
               diff_selected_action, returns_info, hit, suessor

    def reshape(self, obs, actions, rewards, values, states, masks, selected_action,
                diff_selected_action, returns_info, hit, suessor):
        obs = np.reshape(obs, [self.n_env*self.n_steps] + list(obs.shape[2:]))
        actions = np.reshape(actions, [self.n_env*self.n_steps] + list(actions.shape[2:]))
        rewards = np.reshape(rewards, [self.n_env*self.n_steps])
        values = np.reshape(values, [self.n_env*self.n_steps])
        states = np.reshape(states, states.shape)
        masks = np.reshape(masks, [self.n_env*self.n_steps])
        selected_action = np.reshape(selected_action, [self.n_env*self.n_steps])
        diff_selected_action = np.reshape(diff_selected_action, [self.n_env*self.n_steps])
        returns_info = np.reshape(returns_info, [self.n_env*self.n_steps])
        hit = np.reshape(hit, [self.n_env*self.n_steps])
        suessor = np.reshape(suessor, [self.n_env])

        return obs, actions, rewards, values, states, masks, selected_action, \
               diff_selected_action, returns_info, hit, suessor


