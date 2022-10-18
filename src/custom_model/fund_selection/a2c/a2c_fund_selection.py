from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import gym
import numpy as np
import tensorflow as tf
import sys
import logger
from custom_model.fund_selection.common import Scheduler, find_trainable_variables, mse, \
    total_episode_reward_logger, AbstractEnvRunner, explained_variance, tf_util
from custom_model.fund_selection.policies.base_class_fund_selection import ActorCriticRLModel, SetVerbosity, \
    TensorboardWriter
from custom_model.fund_selection.policies.policies_fund_selection import LstmPolicy, ActorCriticPolicy
# from custom_model.policies.policies_fund_selection import LstmPolicy, ActorCriticPolicy
from custom_model.fund_selection.a2c.buffer import Buffer
from custom_model.fund_selection.a2c.short_term_buffer import SBuffer
from custom_model.fund_selection.a2c.step_buffer import StepBuffer
# from util import funTime, consecutive_true
from util import funTime, discount_reward, writeFile, loadFile
import pandas as pd
from gym.spaces import Discrete, Box, MultiDiscrete
import header.fund_selection.RUNHEADER as RUNHEADER
import util

# import matplotlib
# matplotlib.use('cairo')
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class A2C(ActorCriticRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss caculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01,
                 max_grad_norm=RUNHEADER.m_max_grad_norm,
                 learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=True, write_file=False):

        super(A2C, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = RUNHEADER.m_vf_coef
        self.ent_coef = RUNHEADER.m_ent_coef
        self.pi_coef = RUNHEADER.m_pi_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.params = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None
        self.episode_reward = None

        # custom_variables
        self.day_return = None
        self.sussesor_ph = None
        self.selected_action_ph = None
        self.diff_action_ph = None
        self.returns_info_ph = None
        self.hit_ph = None
        self.cur_lr = None
        self.record_tabular = list()
        self.total_example = RUNHEADER.m_total_example
        self.buffer_size = RUNHEADER.m_buffer_size
        self.replay_ratio = RUNHEADER.m_replay_ratio
        self.replay_start = RUNHEADER.m_replay_start
        self.main_replay_start = RUNHEADER.m_main_replay_start

        self.tensorboard_log_update = RUNHEADER.m_tensorboard_log_update

        self.write_file = write_file
        self.fp_short_term_memory = './short_term_memory_log.txt'

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    @funTime('graph building')
    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                               "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                if self.verbose == 2:
                    self.sess = tf_util.debug_session(graph=self.graph, port="localhost:7000", mode=0)
                else:
                    self.sess = tf_util.make_session(graph=self.graph)

                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, LstmPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         n_batch_step, reuse=False, **self.policy_kwargs)

                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                              self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)

                with tf.compat.v1.variable_scope("loss", reuse=False):
                    self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
                    self.entropy = tf.reduce_mean(input_tensor=train_model.proba_distribution.entropy())
                    self.pg_loss = tf.reduce_mean(input_tensor=self.advs_ph * neglogpac)
                    self.vf_loss = mse(tf.squeeze(train_model.value_fn), self.rewards_ph)
                    loss_1 = self.pg_loss * self.pi_coef - self.entropy * self.ent_coef
                    loss_2 = self.vf_loss * self.vf_coef
                    loss = loss_1 + loss_2

                    tf.compat.v1.summary.scalar('entropy_loss', self.entropy)
                    tf.compat.v1.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.compat.v1.summary.scalar('value_function_loss', self.vf_loss)
                    tf.compat.v1.summary.scalar('value_function', tf.reduce_mean(input_tensor=train_model.value_fn))
                    tf.compat.v1.summary.scalar('advs', tf.reduce_mean(input_tensor=self.advs_ph))
                    tf.compat.v1.summary.scalar('neglogpac', tf.reduce_mean(input_tensor=neglogpac))
                    tf.compat.v1.summary.scalar('loss', loss)
                    tf.compat.v1.summary.scalar('loss_1', loss_1)
                    tf.compat.v1.summary.scalar('loss_2', loss_2)

                    self.params = find_trainable_variables("model")
                    grads = tf.gradients(ys=loss, xs=self.params)
                    grads_pi = tf.gradients(ys=loss_1, xs=self.params)
                    grads_vf = tf.gradients(ys=loss_2, xs=self.params)
                    if self.max_grad_norm is not None:
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        grads_pi, _ = tf.clip_by_global_norm(grads_pi, self.max_grad_norm)
                        grads_vf, _ = tf.clip_by_global_norm(grads_vf, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                    grads_pi = list(zip(grads_pi, self.params))
                    grads_vf = list(zip(grads_vf, self.params))
                    # Added 2019-08-27
                    merge_grads = list()
                    for idx in range(len(self.params)):
                        param = self.params[idx]
                        if (param.name == 'model/pi/w_0') or (param.name == 'model/pi/b_0'):
                            merge_grads.append(grads_pi[idx])
                        elif (param.name == 'model/vf/w_0') or (param.name == 'model/vf/b_0'):
                            merge_grads.append(grads_vf[idx])
                        else:
                            merge_grads.append(grads[idx])
                    grads = merge_grads

                # Todo: check it later
                with tf.compat.v1.variable_scope("weight_visualization", reuse=False):
                    # Todo: check it later
                    for partial_deviation, param in grads:
                        if partial_deviation is not None:
                            tf.compat.v1.summary.scalar('partial_dev_' + param.name,
                                              tf.reduce_mean(input_tensor=partial_deviation))
                            tf.compat.v1.summary.scalar(param.name, tf.reduce_mean(input_tensor=param))
                            tf.compat.v1.summary.histogram('partial_dev_' + param.name, partial_deviation)
                            tf.compat.v1.summary.histogram(param.name, param)

                with tf.compat.v1.variable_scope("step_info", reuse=False):
                    self.sussesor_ph = tf.compat.v1.placeholder(tf.int32, [None], name="sussesor_ph")
                    self.selected_action_ph = tf.compat.v1.placeholder(tf.int32, [None], name="selected_action_ph")
                    self.diff_action_ph = tf.compat.v1.placeholder(tf.int32, [None], name="diff_action_ph")
                    self.returns_info_ph = tf.compat.v1.placeholder(tf.float32, [None], name="returns_info_ph")
                    self.hit_ph = tf.compat.v1.placeholder(tf.float32, [None], name="hit_ph")
                    tf.compat.v1.summary.scalar('sussesor', tf.reduce_mean(input_tensor=self.sussesor_ph))
                    tf.compat.v1.summary.scalar('num_action', tf.reduce_mean(input_tensor=self.selected_action_ph))
                    tf.compat.v1.summary.scalar('diff_action', tf.reduce_mean(input_tensor=self.diff_action_ph))
                    tf.compat.v1.summary.scalar('5_days_returns', tf.reduce_mean(input_tensor=self.returns_info_ph))
                    tf.compat.v1.summary.scalar('hit', tf.reduce_mean(input_tensor=self.hit_ph))

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.compat.v1.summary.scalar('discounted_rewards', tf.reduce_mean(input_tensor=self.rewards_ph))
                    tf.compat.v1.summary.scalar('learning_rate', tf.reduce_mean(input_tensor=self.learning_rate_ph))
                    tf.compat.v1.summary.scalar('advantage', tf.reduce_mean(input_tensor=self.advs_ph))
                    if self.full_tensorboard_log:
                        tf.compat.v1.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.compat.v1.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.compat.v1.summary.histogram('advantage', self.advs_ph)
                        # if tf_util.is_image(self.observation_space):
                        #     tf.summary.image('observation', train_model.obs_ph)
                        # else:
                        #     tf.summary.histogram('observation', train_model.obs_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.compat.v1.summary.image('processed_observation', train_model.processed_obs)
                        else:
                            tf.compat.v1.summary.histogram('processed_observation', train_model.processed_obs)
                            tf.compat.v1.summary.scalar('processed_observation', tf.reduce_mean(input_tensor=train_model.processed_obs))

                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)  # add for batch normalization - 2019-08-07
                with tf.control_dependencies(update_ops):  # add for batch normalization - 2019-08-07
                    trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                        epsilon=self.epsilon)
                self.apply_backprop = trainer.apply_gradients(grads)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.compat.v1.global_variables_initializer().run(session=self.sess)

                self.summary = tf.compat.v1.summary.merge_all()

    def _train_step(self, obs, states, rewards, masks, actions, values, update, writer=None,
                    suessor=None, selected_action=None, diff_selected_action=None, returns_info=None, hit=None,
                    replay=False):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        if RUNHEADER.m_online_buffer:
            self.cur_lr = None
            for _ in range(len(obs)):
                self.cur_lr = self.learning_rate_schedule.value()
            assert self.cur_lr is not None, "Error: the observation input array cannon be empty"
        else:  # fix learning rate for offline learning
            self.cur_lr = RUNHEADER.m_offline_learning_rate

        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: self.cur_lr,
                  self.sussesor_ph: suessor, self.selected_action_ph: selected_action,
                  self.diff_action_ph: diff_selected_action,
                  self.returns_info_ph: returns_info, self.hit_ph: hit}

        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % self.tensorboard_log_update == 0:
                run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop],
                    td_map, options=run_options, run_metadata=run_metadata)
                #  Add for replay memory
                if not replay:
                    writer.add_run_metadata(run_metadata, 'step%d' % (update * (self.n_batch + 1)))
            else:
                summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)
            if not replay:
                writer.add_summary(summary, update * (self.n_batch + 1))

        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)

        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=10, tb_log_name="A2C",
              reset_num_timesteps=True, model_location=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)
            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
            self.episode_reward = np.zeros((self.n_envs,))
            self.day_return = np.zeros((self.n_envs, 5,))  # 0, 1, 5, 10, 20

            if RUNHEADER.m_online_buffer:  # generate buffer
                buffer = Buffer(env=self.env, n_steps=self.n_steps, size=self.buffer_size)
            else:  # load offline-buffer and continue stack buffer
                print_out_csv = list()
                for epoch in range(RUNHEADER.m_offline_learning_epoch):
                    file_names = os.listdir(RUNHEADER.m_offline_buffer_file)
                    buffer_names = [buffer_name for buffer_name in file_names if '.pkl' and 'buffer_' in buffer_name]
                    if epoch == 0:
                        # first drops non-optimized model
                        self.save('{}/fs_epoch_init_{}.pkl'.format(model_location, seed))
                    for buffer_name in buffer_names:
                        buffer = None
                        buffer = loadFile(RUNHEADER.m_offline_buffer_file + '/' + buffer_name[:-4])
                        print_out_csv = self.offline_learn(writer, runner, buffer, model_location, epoch,
                                                           print_out_csv)
                        pd.DataFrame(print_out_csv, columns=['epoch', 'update', 'policy_entropy', 'policy_loss',
                                                             'value_loss', 'rewards', 'values']). \
                            to_csv('{}/record_tabular_epoch_summary.csv'.format(model_location))
                self.record_tabular = list()
                #  break after offline learning
                exit()

            # first drops non-optimized model
            self.save('{}/fs_epoch_init_{}.pkl'.format(model_location, seed))

            # if True:
            #     exit()

            t_start = time.time()
            print_rewards = list()
            current_timesteps = 0
            current_episode_idx = 0
            ep_dsr = list()
            ep_dones = list()
            entry_th = None
            mask_th = None
            learned_experiences_ep = 0
            non_pass_episode = 0
            short_term_roll_out = 0
            delete_repeat_sample = 0  # 5times repeat
            for update in range(1, total_timesteps // self.n_batch + 1):
                if update == RUNHEADER.m_total_example:
                    exit()
                # print('\n')
                """ Training
                """
                # 1. set current observation episode
                if np.sum(self.env.get_attr('eoe')):
                    current_episode_idx = 0
                self.env.set_attr('current_episode_idx', current_episode_idx)
                # self.env.env_method('clear_cache')

                # 2. gather simulation result
                bool_runner = True
                max_iter = 0
                cond_cnt_1 = 0
                cond_cnt_2 = 0
                exit_cnt = 0
                current_iter_cnt = 0
                n_pushed_experiences = 0
                replay_iteration = RUNHEADER.m_replay_iteration

                if update < int(self.total_example / 60):  # first 5% of data
                    entry_th = RUNHEADER.m_entry_th
                    mask_th = RUNHEADER.m_mask_th
                else:
                    entry_th = np.mean(ep_dsr) + np.std(ep_dsr) * RUNHEADER.improve_th
                    mask_th = np.mean(ep_dones) + np.std(ep_dones) * RUNHEADER.improve_th
                    if entry_th < RUNHEADER.m_min_entry_th: entry_th = RUNHEADER.m_min_entry_th
                    if mask_th < RUNHEADER.m_min_mask_th: mask_th = RUNHEADER.m_min_mask_th
                    if entry_th > RUNHEADER.m_max_entry_th: entry_th = RUNHEADER.m_max_entry_th
                    if mask_th > RUNHEADER.m_max_mask_th: mask_th = RUNHEADER.m_max_mask_th

                # gather
                sys.stdout.write('\n')
                sys.stdout.flush()
                short_term_buffer = SBuffer(env=self.env, n_steps=self.n_steps)
                while bool_runner:
                    self.env.env_method('clear_cache')
                    # step memory
                    obs, states, rewards, masks, actions, values, true_reward, \
                    info, get_progress_info, suessor, selected_action, \
                    diff_selected_action, returns_info, hit = runner.run()

                    assert not ((len(np.argwhere(selected_action > self.env.action_space.shape)) > 0)
                                or (len(np.argwhere(selected_action < 0)) > 0)), 'check selected action!!!'

                    # short-term memory
                    bool_runner, entry_th, mask_th, cond_cnt_1, cond_cnt_2, max_iter, exit_cnt, short_term_buffer = \
                        self.short_term_simulation(rewards, masks, entry_th, mask_th, cond_cnt_1, cond_cnt_2,
                                                   max_iter, bool_runner, exit_cnt, short_term_buffer,
                                                   obs, actions, values, states, suessor, selected_action,
                                                   diff_selected_action, returns_info, hit, current_iter_cnt)
                    sys.stdout.write('\r>> [%d] short-term memory:  %d/%d' % (current_iter_cnt,
                                                                              short_term_buffer.num_in_buffer,
                                                                              self.n_envs))
                    sys.stdout.flush()
                    current_iter_cnt = current_iter_cnt + 1
                # gather end

                if short_term_buffer.num_in_buffer > 0:
                    # caution: this is a short-term simulation roll-out not a step-simulation one
                    short_term_roll_out = int(self.n_envs - short_term_buffer.num_in_buffer)
                    """ to calculate mean-variance of experience
                    """
                    # mark 1.
                    obs, actions, rewards, values, states, masks, suessor, selected_action, \
                    diff_selected_action, returns_info, hit = short_term_buffer.get()

                    ep_dsr.append(float(np.mean(np.reshape(rewards, [self.n_envs, self.n_steps]))))
                    ep_dones.append(float(np.mean(suessor)))
                else:  # forget experience - when fails for all tries (m_max_iter*m_exit_cnt iteration)
                    ep_dsr = list((np.array(ep_dsr) - np.array(ep_dsr) * RUNHEADER.m_forget_experience))
                    ep_dones = list((np.array(ep_dones) - np.array(ep_dones) * RUNHEADER.m_forget_experience))
                current_episode_idx = current_episode_idx + 1
                # delete_repeat_sample = delete_repeat_sample + 1  # Delete Clear cache
                # if (delete_repeat_sample % 1000) == 0:
                #     current_episode_idx = current_episode_idx + 1

                # 3. add gathered experiences to buffer
                # if buffer is not None:
                if (buffer is not None) and (short_term_buffer.num_in_buffer > 0):
                    buffer.put(obs, actions, rewards, values, states, masks,
                               suessor, selected_action, diff_selected_action, returns_info, hit)

                # 4. re-call experiences [sub-process of replay]
                if buffer is not None and self.replay_ratio > 0 and buffer.has_atleast(self.n_envs):
                    if short_term_buffer.num_in_buffer == 0:
                        obs, actions, rewards, values, states, masks, suessor, selected_action, \
                        diff_selected_action, returns_info, hit, learned_experiences_ep, non_pass_episode = \
                            self.re_call_buff(buffer, learned_experiences_ep, non_pass_episode, runner)

                # 5. fit training model
                policy_loss, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values,
                                                                           self.num_timesteps // (self.n_batch + 1),
                                                                           writer,
                                                                           suessor, selected_action,
                                                                           diff_selected_action,
                                                                           returns_info, hit)

                """ Trace actions
                """
                tmp = np.reshape(actions, [self.n_envs, self.n_steps, actions.shape[-1]])[0].T
                plt.imsave('./save/images/{}_{}_{}.jpeg'.format(str(delete_later), update,
                                                                int(np.mean(np.sum(tmp, axis=0)))), tmp * 255)

                """Blows describe post-processes during the training
                """
                # model save according to the time stamps
                current_timesteps = current_timesteps + 1
                if update % int(self.total_example // 200) == 0:  # drop a model with every 0.5% of examples
                    self.save('{}/fs_{}_ev{:3.3}_pe{:3.3}_pl{:3.3}_vl{:3.3}.pkl'.format(model_location,
                                                                                        current_timesteps,
                                                                                        explained_var, policy_entropy,
                                                                                        policy_loss,
                                                                                        value_loss))
                # buffer save
                if (buffer.num_in_buffer >= buffer.size) or (update == RUNHEADER.m_total_example - 1):
                    writeFile('{}/buffer_E{}_S{}_U{}'.format(model_location, self.n_envs, self.n_steps, update), buffer)
                    buffer = None
                    buffer = Buffer(env=self.env, n_steps=self.n_steps, size=self.buffer_size)  # re-init

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)
                self.num_timesteps += self.n_batch + 1

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                """Write printout and save csv
                """
                explained_var = self.print_out_tabular(update=update, values=values, rewards=rewards,
                                                       policy_entropy=policy_entropy, policy_loss=policy_loss,
                                                       value_loss=value_loss,
                                                       diff_selected_action=diff_selected_action, hit=hit,
                                                       suessor=suessor, selected_action=selected_action,
                                                       model_location=model_location, entry_th=entry_th,
                                                       mask_th=mask_th, buffer=buffer,
                                                       learned_experiences_ep=learned_experiences_ep,
                                                       non_pass_episode=non_pass_episode,
                                                       short_term_roll_out=short_term_roll_out,
                                                       log_interval=log_interval)
                """Restore extra information
                """
                print_rewards.append([float(np.mean(np.reshape(rewards, [self.n_envs, self.n_steps]))),
                                      float(np.mean(np.reshape(true_reward, [self.n_envs, self.n_steps])))])

                """ Replay [main-process of replay]
                """
                if (self.replay_ratio > 0) and buffer.has_atleast(self.replay_start) \
                        and (update > self.main_replay_start):
                    samples_number = np.random.poisson(self.replay_ratio)
                    for _ in range(samples_number * replay_iteration):
                        # get obs, actions, rewards, mus, dones from buffer.
                        obs, states, rewards, masks, actions, values, true_reward, info, get_progress_info, suessor, \
                        selected_action, diff_selected_action, returns_info, hit = \
                            self.retrive_experiences(buffer, runner)

                        self._train_step(obs, states, rewards, masks, actions, values,
                                         self.num_timesteps // (self.n_batch + 1),
                                         writer,
                                         suessor, selected_action,
                                         diff_selected_action,
                                         returns_info, hit, replay=True)
                    learned_experiences_ep = learned_experiences_ep + samples_number

            pd.DataFrame(data=np.array(print_rewards), columns=['rewards', 'true_rewards']). \
                to_csv('{}/train_rewards_records.csv'.format(model_location))

        return self

    def retrive_experiences(self, buffer, runner):
        # get obs, actions, rewards, mus, dones from buffer.
        obs_buffer, action_buffer, rewards_buffer, _, _, dones_buffer, _, _, \
        _, return_info_buffer, _ = buffer.get()
        return runner.run(obs_buffer=obs_buffer, action_buffer=action_buffer,
                          rewards_buffer=rewards_buffer, dones_buffer=dones_buffer,
                          return_info_buffer=return_info_buffer, online_buffer=False)

    def offline_learn(self, writer, runner, buffer, model_location, epoch, print_out_csv):
        samples_number = int(buffer.num_in_buffer / self.n_envs)
        values_summary, rewards_summary, policy_entropy_summary, policy_loss_summary, value_loss_summary \
            = list(), list(), list(), list(), list()
        rewards, values = None, None

        for update in range(samples_number):  # learning for 1 epoch

            obs, states, rewards, masks, actions, values, true_reward, info, get_progress_info, suessor, \
            selected_action, diff_selected_action, returns_info, hit = \
                self.retrive_experiences(buffer, runner)

            timestamp = epoch * samples_number + update
            policy_loss, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values,
                                                                       timestamp,
                                                                       # self.num_timesteps // (self.n_batch + 1),
                                                                       writer,
                                                                       suessor, selected_action,
                                                                       diff_selected_action,
                                                                       returns_info, hit, replay=False)
            policy_entropy_summary.append(policy_entropy)
            policy_loss_summary.append(policy_loss)
            value_loss_summary.append(value_loss)

            if update % 250 == 0:
                tmp = np.reshape(actions, [self.n_envs, self.n_steps, actions.shape[-1]])[0].T
                plt.imsave('{}/actions_{}_{}_{}.jpeg'.format(model_location,
                                                             epoch, update, int(np.mean(np.sum(tmp, axis=0)))),
                           tmp * 255)

                # Drop the model
                self.save('{}/fs_epoch_{}_{}_pe{:3.3}_pl{:3.3}_vl{:3.3}.pkl'.format(model_location, epoch, update,
                                                                                    policy_entropy,
                                                                                    policy_loss,
                                                                                    value_loss))

            sys.stdout.write('\r>> [{}] Offline Learning step: {}/{}'.format(epoch, update, samples_number))
            sys.stdout.flush()

            _ = self.print_out_tabular(policy_entropy=policy_entropy,
                                       policy_loss=policy_loss,
                                       value_loss=value_loss, model_location=model_location,
                                       rewards=rewards, values=values,
                                       opt=1)
            print_out_csv.append([epoch, update, np.mean(policy_entropy_summary), np.mean(policy_loss_summary),
                                  np.mean(value_loss_summary),
                                  float(np.mean(np.reshape(rewards, [self.n_envs, self.n_steps]))),
                                  float(np.mean(np.reshape(values, [self.n_envs, self.n_steps])))])

        _ = self.print_out_tabular(policy_entropy=np.mean(policy_entropy_summary),
                                   policy_loss=np.mean(policy_loss_summary),
                                   value_loss=np.mean(value_loss_summary), model_location=model_location,
                                   rewards=rewards, values=values,
                                   opt=1)
        return print_out_csv

    def re_call_buff(self, buffer, learned_experiences_ep, non_pass_episode, runner):

        obs, states, rewards, masks, actions, values, true_reward, info, get_progress_info, suessor, \
        selected_action, diff_selected_action, returns_info, hit = \
            self.retrive_experiences(buffer, runner)

        learned_experiences_ep = learned_experiences_ep + 1
        non_pass_episode = non_pass_episode + 1

        return obs, actions, rewards, values, states, masks, suessor, selected_action, \
               diff_selected_action, returns_info, hit, learned_experiences_ep, non_pass_episode

    # @funTime('save model')
    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    def _short_term_simulation(self, rewards, masks, hit, actions, entry_th, mask_th):
        # rewards
        cond = np.reshape(rewards, [self.n_envs, self.n_steps])
        cond_1_idx = np.squeeze(np.argwhere(np.sum(cond, axis=1) > entry_th), axis=1).tolist()
        cond_1 = len(cond_1_idx)

        # dones
        cond_2 = np.reshape(masks, [self.n_envs, self.n_steps])
        cond_2_idx = np.squeeze(np.argwhere(np.array([util.consecutive_true(item) for item in cond_2]) > mask_th),
                                axis=1).tolist()
        cond_2 = len(cond_2_idx)

        # disjunction of rewards and dones
        _env_idx = np.array(list(set(cond_1_idx + cond_2_idx)))

        # hit rates
        cond_3_idx = np.squeeze(np.argwhere(
            np.mean(np.reshape(hit, [self.n_envs, self.n_steps]), axis=1) >= 0.62), axis=1).tolist()
        cond_3 = len(cond_3_idx)

        # actions
        tmp_action = np.sum(np.reshape(actions, [self.n_envs, self.n_steps, actions.shape[-1]]), axis=2)
        tmp_action = np.where(tmp_action > RUNHEADER.m_allow_actions_max, 0, tmp_action)
        tmp_action = np.where(tmp_action < RUNHEADER.m_allow_actions_min, 0, tmp_action)
        cond_4_idx = np.squeeze(np.argwhere(np.min(tmp_action, axis=1) > 0), axis=1).tolist()
        cond_4 = len(cond_4_idx)

        if self.write_file:
            fp_short_term_memory = open(self.fp_short_term_memory, 'a')
            log_short_memory = '{},{},{},{}'.format(cond_1, cond_2, cond_3, cond_4)
            print(log_short_memory, file=fp_short_term_memory)
            fp_short_term_memory.close()

        env_idx = list()
        for idx in _env_idx:
            #  enable for now
            if (idx in cond_3_idx) and (idx in cond_4_idx):
                env_idx.append(idx)

        return cond_1, cond_2, cond_3, cond_4, env_idx

    def short_term_simulation(self, rewards, masks, entry_th, mask_th, cond_cnt_1, cond_cnt_2, max_iter,
                              bool_runner, exit_cnt, short_term_buffer,
                              obs, actions, values, states, suessor, selected_action,
                              diff_selected_action, returns_info, hit, current_iter_cnt):

        #  store examples to short-term memory
        cond_1, cond_2, cond_3, cond_4, env_idx = self._short_term_simulation(rewards, masks, hit,
                                                                              actions, entry_th, mask_th)
        short_term_buffer.put(obs, actions, rewards, values, states, masks, suessor, selected_action,
                              diff_selected_action, returns_info, hit, env_idx)

        #  adjust threshold
        if short_term_buffer.num_in_buffer == 0:
            sys.stdout.write('\n[Empty experience] Calculate threshold reference. [{} / {}] '
                             'entry_th: {}, mask_th: {}\n'.format(cond_1, cond_2, entry_th, mask_th))

            sys.stdout.flush()
            if cond_1 == cond_2:
                cond_cnt_1 = cond_cnt_1 + 1
                cond_cnt_2 = cond_cnt_2 + 1
            else:
                if cond_1 < cond_2:
                    cond_cnt_1 = cond_cnt_1 + 1
                else:
                    cond_cnt_2 = cond_cnt_2 + 1

        # control roof parameters and adjust threshold for sampling
        if max_iter == RUNHEADER.m_max_iter:  # modify threshold
            exit_cnt = exit_cnt + 1
            if cond_1 == cond_2:
                entry_th = entry_th - (entry_th * RUNHEADER.m_forget_experience_short_term)
                mask_th = mask_th - (mask_th * RUNHEADER.m_forget_experience_short_term)
            else:
                if cond_cnt_1 > cond_cnt_2:
                    entry_th = entry_th - (entry_th * RUNHEADER.m_forget_experience_short_term)
                else:
                    mask_th = mask_th - (mask_th * RUNHEADER.m_forget_experience_short_term)

            if entry_th < RUNHEADER.m_min_entry_th:
                entry_th = RUNHEADER.m_min_entry_th
            if mask_th < RUNHEADER.m_min_mask_th:
                mask_th = RUNHEADER.m_min_mask_th

            sys.stdout.write('\n[Hard threshold] Reduce entry and mask threshold!!! -> '
                             'entry_th: {}, mask_th: {}'.format(entry_th, mask_th))
            sys.stdout.flush()

            max_iter = 0  # caution .. Todo: code refactoring
            cond_cnt_1 = 0
            cond_cnt_2 = 0
        max_iter = max_iter + 1

        # exit condition
        if exit_cnt == RUNHEADER.m_exit_cnt:
            bool_runner = False
        if short_term_buffer.num_in_buffer == self.n_envs:
            bool_runner = False
        # early-stop for short-term memory
        # if (short_term_buffer.num_in_buffer > RUNHEADER.m_early_stop) and (current_iter_cnt > 1):
        if short_term_buffer.num_in_buffer > RUNHEADER.m_early_stop:
            bool_runner = False

        return bool_runner, entry_th, mask_th, cond_cnt_1, cond_cnt_2, max_iter, exit_cnt, short_term_buffer

    def print_out_tabular(self, update=None, values=None, rewards=None, policy_entropy=None,
                          policy_loss=None, value_loss=None, diff_selected_action=None,
                          hit=None, suessor=None, selected_action=None, model_location=None,
                          opt=0, entry_th=None, mask_th=None, buffer=None, learned_experiences_ep=None,
                          non_pass_episode=None, short_term_roll_out=None, log_interval=None):
        """Write printout and save csv
        """
        if opt == 0:  # default tabular
            tabular_column_name = ['n_updates', 'total_timesteps', 'learning_rate', 'policy_entropy',
                                   'policy_loss', 'value_loss', 'explained_variance',
                                   'Episode_diff_selected_action', 'Episode_hit', 'Episode_Dones',
                                   'Episode_Rewards', 'Episode_actions', 'values', 'entry_th', 'mask_th',
                                   'current_buff_size', 'learned_replay_ep',
                                   'non_pass_episode', 'date', 'short_term_roll_out']
            explained_var = explained_variance(values, rewards)
            self.record_tabular.append([str(update) + '/' + str(RUNHEADER.m_total_example), self.num_timesteps,
                                        self.cur_lr, float(policy_entropy),
                                        float(policy_loss), float(value_loss), float(explained_var),
                                        int(np.mean(
                                            np.mean(np.reshape(diff_selected_action, [self.n_envs, self.n_steps]),
                                                    axis=1))),
                                        float(
                                            np.mean(np.mean(np.reshape(hit, [self.n_envs, self.n_steps]), axis=1))),
                                        float(np.mean(suessor)),
                                        float(np.mean(np.reshape(rewards, [self.n_envs, self.n_steps]))),
                                        int(np.mean(
                                            np.mean(np.reshape(selected_action, [self.n_envs, self.n_steps])))),
                                        float(np.mean(np.reshape(values, [self.n_envs, self.n_steps]))),
                                        float(entry_th), float(mask_th), int(buffer.num_in_buffer),
                                        learned_experiences_ep, non_pass_episode,
                                        delete_later, short_term_roll_out])
            if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                print('\n')
                print_out_tabular = self.record_tabular[-1]
                for idx in range(len(tabular_column_name)):
                    logger.record_tabular(tabular_column_name[idx], print_out_tabular[idx])
                logger.dump_tabular()
            pd.DataFrame(data=np.array(self.record_tabular), columns=tabular_column_name). \
                to_csv('{}/record_tabular.csv'.format(model_location))
        else:  # tabular print for offline learning
            tabular_column_name = ['learning_rate', 'policy_entropy',
                                   'policy_loss', 'value_loss', 'explained_variance', 'loss', 'values', 'rewards',
                                   'advs', 'neglogp']
            # explained_var = 1 - (np.var(rewards - values) / np.var(rewards))
            explained_var = explained_variance(values, rewards)
            values = float(np.mean(np.reshape(values, [self.n_envs, self.n_steps])))
            rewards = float(np.mean(np.reshape(rewards, [self.n_envs, self.n_steps])))
            self.record_tabular.append([self.cur_lr, float(policy_entropy), float(policy_loss),
                                        float(value_loss), float(explained_var),
                                        (policy_loss - policy_entropy *
                                         RUNHEADER.m_ent_coef + value_loss * RUNHEADER.m_vf_coef),
                                        float(values), float(rewards),
                                        float(values - rewards), float(policy_loss / (values - rewards))
                                        ])
            if self.verbose >= 1:
                print('\n')
                print_out_tabular = self.record_tabular[-1]
                for idx in range(len(tabular_column_name)):
                    logger.record_tabular(tabular_column_name[idx], print_out_tabular[idx])
                logger.dump_tabular()
            pd.DataFrame(data=np.array(self.record_tabular), columns=tabular_column_name). \
                to_csv('{}/record_tabular_buffer.csv'.format(model_location))

        return explained_var


class A2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(A2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma
        self.n_env = n_env = env.num_envs
        if isinstance(env.action_space, Discrete):
            self.n_act = env.action_space.n
        elif isinstance(env.action_space, MultiDiscrete):
            self.n_act = env.action_space.shape[-1]
        else:
            self.n_act = env.action_space.shape[-1]
        self.n_batch = n_env * n_steps

    def run(self, obs_buffer=None, action_buffer=None, rewards_buffer=None, dones_buffer=None,
            return_info_buffer=None, online_buffer=True):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        print('\n')
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        previous_action = np.zeros(1)
        tmp_info = list()
        # Todo: assume that each episodes are independent (it depends on your perspective to given problem)
        self.states = self.init_state
        mb_states = self.states
        self.dones = [False for _ in range(self.n_env)]
        selected_action = list()
        diff_selected_action = list()
        returns_info = list()
        hit = list()

        # step simulation
        for n_steps_idx in range(self.n_steps):
            if online_buffer:  # on-line simulation
                step_memory = self.step_simulation(n_steps_idx=n_steps_idx)
                actions, values, states, obs, rewards, dones, info = step_memory.get()
            else:  # offline-learning or re-call experiences for online simulation
                actions, values, states, obs, rewards, dones, tmp_log = \
                    self.step_simulation_offline(n_steps_idx=n_steps_idx, obs_buffer=obs_buffer,
                                                 action_buffer=action_buffer,
                                                 rewards_buffer=rewards_buffer,
                                                 dones_buffer=dones_buffer,
                                                 return_info_buffer=return_info_buffer)

            selected_action.append(np.sum(actions, axis=1).tolist())
            diff_selected_action.append(np.sum(abs(previous_action - actions), axis=1).tolist())
            previous_action = actions
            mb_obs.append(np.copy(self.obs))
            # mb_states.append(np.copy(self.states))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            self.states = states
            self.dones = dones
            # self.obs = obs
            mb_rewards.append(rewards)

            if online_buffer:
                tmp_info.append(info)
                tmp_log = [tmp['5day_return'] for tmp in info]
                returns_info.append(tmp_log)
                hit.append(np.where(np.array(tmp_log) > 0, 1, 0))
            else:
                returns_info.append(tmp_log.tolist())
                hit.append(np.where(tmp_log > 0, 1, 0))
        # stack step experiences
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()

        if online_buffer:
            # discount/bootstrap off value fn.. backward continuous reward sum
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    # rewards = discount_reward(rewards + [value], dones + [0], RUNHEADER.m_discount_factor)[:-1]
                    rewards = discount_reward(rewards, dones, RUNHEADER.m_discount_factor)
                else:
                    rewards = discount_reward(rewards, dones, RUNHEADER.m_discount_factor)
                mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])

        suessor = [util.consecutive_true(item) for item in np.reshape(mb_masks, [self.n_env, self.n_steps])]
        get_progress_info = self.env.env_method('get_progress_info')

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, true_rewards, \
               tmp_info, get_progress_info, suessor, \
               np.reshape(np.array(selected_action).T, [self.n_env * self.n_steps]), \
               np.reshape(np.array(diff_selected_action).T, [self.n_env * self.n_steps]), \
               np.reshape(np.array(returns_info).T, [self.n_env * self.n_steps]), \
               np.reshape(np.array(hit).T, [self.n_env * self.n_steps])

    def step_simulation(self, clipped_actions=0, n_steps_idx=None):
        search_cnt = 0
        terminate_cnt = 0
        min_reward = 0
        b_step = True
        mode = False
        disable_dones = False
        sample_th = RUNHEADER.m_sample_th
        step_memory = StepBuffer(env=self.env)
        self.env.set_attr('current_step', n_steps_idx)
        # if n_steps_idx == 1:
        #     print('test')
        self.obs = np.array(self.env.env_method('obs_from_env'))
        while b_step:
            if mode:
                sample_th = np.inf

            """get step example and stack to step memory
            """
            actions, values, states, neglogp = self.model.step(self.obs, self.states, self.dones, sample_th=sample_th)

            # step actions - for display
            diff_action = np.sum(np.where((actions - clipped_actions) == 0, 0, 1), axis=1)
            clipped_actions = actions
            num_clipped_actions = np.sum(clipped_actions, axis=1)
            change_rate = float(np.mean(diff_action / (num_clipped_actions + 1E-15)))
            changes = float(np.mean(diff_action))

            # clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # get rewards, dones, info and next observation
            obs, rewards, dones, info = self.env.step(clipped_actions)

            assert np.allclose(self.obs, obs), 'Delete this code later, just for code test!!!'

            # stack an experience
            step_memory.put(actions, values, states, obs, rewards, dones, info,
                            mode=mode, min_reward=min_reward, disable_dones=disable_dones)

            """get sample_th and mode corresponding its search iteration 
            """
            if step_memory.num_in_buffer == self.n_env:
                b_step = False
            else:
                if terminate_cnt < 1:  # once search examples with mode=False
                    sample_th, terminate_cnt, search_cnt = \
                        self.get_sample_th(search_cnt, terminate_cnt, step_memory, actions)
                else:  # mode=True case
                    # retry sampling until step_memory.num_in_buffer > 0 with mode=True
                    if (terminate_cnt == 1) and (disable_dones == False):
                        search_cnt = 0
                        disable_dones = True
                    else:
                        if (terminate_cnt > 1) and (step_memory.num_in_buffer > 0):
                            b_step = False  # roll-out examples
                        else:
                            if not mode:  # init search_cnt for mode=True search
                                search_cnt = 0
                                mode = True
                            min_reward = min_reward - 0.02

                    sample_th, terminate_cnt, search_cnt = \
                        self.get_sample_th(search_cnt, terminate_cnt, step_memory, actions)

            num_in_buffer = step_memory.num_in_buffer
            sys.stdout.write('\r>> [%s][%s][%d-%d] step memory:  %d, %d/%d, %3.2f(%3.2f), s_th:%3.2f' % (
                info[0]['date'],
                terminate_cnt,
                n_steps_idx, int(np.mean(np.sum(actions, axis=1))),
                search_cnt, num_in_buffer, self.n_env,
                int(changes), change_rate, sample_th))
            sys.stdout.flush()

            # delete later
            global delete_later, delete_later2
            delete_later = info[0]['date']
            delete_later2 = sample_th

        return step_memory

    def step_simulation_offline(self, n_steps_idx=None, obs_buffer=None, action_buffer=None,
                                rewards_buffer=None, dones_buffer=None, return_info_buffer=None):
        sample_th = RUNHEADER.m_sample_th
        """get step example and stack to step memory
        """
        self.obs = obs_buffer[n_steps_idx]
        _, values, states, _ = self.model.step(self.obs, self.states, self.dones, sample_th=sample_th)
        actions = action_buffer[n_steps_idx]

        # get rewards, dones, info and next observation
        obs, rewards, dones, return_info = self.obs, rewards_buffer[n_steps_idx], \
                                           dones_buffer[n_steps_idx], return_info_buffer[n_steps_idx]

        return actions, values, states, obs, rewards, dones, return_info

    def get_sample_th(self, search_cnt, terminate_cnt, step_memory, actions):
        sample_th = None
        """Get current sample_th
        """
        if search_cnt <= RUNHEADER.m_interval[0]:
            sample_th = RUNHEADER.m_sample_th
        elif search_cnt >= RUNHEADER.m_interval[-1]:
            sample_th = RUNHEADER.m_interval_value[-1]
        else:
            for idx in range(len(RUNHEADER.m_interval)):
                if idx > 0:
                    if (search_cnt > RUNHEADER.m_interval[idx - 1]) and (search_cnt < RUNHEADER.m_interval[idx]):
                        sample_th = RUNHEADER.m_interval_value[idx - 1]

        """Control parameter setting according to the sample_th
        """
        if sample_th == -np.inf:  # init for re-iteration
            sample_th = RUNHEADER.m_sample_th
            terminate_cnt = terminate_cnt + 1
            if step_memory.num_in_buffer >= RUNHEADER.m_early_stop:
                terminate_cnt = np.inf

        """Dynamic sampling
        """
        if sample_th != np.inf:
            if int(np.mean(np.sum(actions, axis=1))) > RUNHEADER.m_target_actions:
                if sample_th < 0.002:
                    sample_th = 0.002
                sample_th = sample_th - 0.001
            elif int(np.mean(np.sum(actions, axis=1))) < RUNHEADER.m_target_actions:
                if sample_th > 0.90:
                    sample_th = 0.90
                sample_th = sample_th + 0.001
            RUNHEADER.m_sample_th = sample_th

        search_cnt = search_cnt + 1
        return sample_th, terminate_cnt, search_cnt


if __name__ == '__main__':
    None
