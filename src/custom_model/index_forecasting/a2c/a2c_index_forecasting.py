from __future__ import absolute_import, division, print_function

import datetime
import os
import sys
import time
from collections import OrderedDict

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from gym.spaces import Box, Discrete, MultiDiscrete
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

import logger
import plot_util
import util
from custom_model.index_forecasting.a2c.buffer import Buffer
from custom_model.index_forecasting.a2c.short_term_buffer import SBuffer
from custom_model.index_forecasting.a2c.step_buffer import StepBuffer
from custom_model.index_forecasting.common import (
    AbstractEnvRunner,
    Scheduler,
    explained_variance,
    find_moving_mean_varience,
    find_shared_variables,
    find_trainable_variables,
    mse,
    ortho_init,
    tf_util,
    total_episode_reward_logger,
)
from custom_model.index_forecasting.policies.base_class_index_forecasting import (
    ActorCriticRLModel,
    SetVerbosity,
    TensorboardWriter,
)
from custom_model.index_forecasting.policies.policies_index_forecasting import (
    ActorCriticPolicy,
    LstmPolicy,
)
from header.index_forecasting import RUNHEADER
from util import discount_reward, funTime, loadFile, writeFile


def type_of_buffer(env, n_steps, buffer_size, is_chunk):
    ratio = 0.20
    buffer_size = int(buffer_size * ratio) if bool(is_chunk) else buffer_size

    return Buffer(env=env, n_steps=n_steps, size=buffer_size)


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

    def __init__(
        self,
        policy,
        env,
        gamma=0.99,
        n_steps=5,
        vf_coef=0.25,
        ent_coef=0.01,
        max_grad_norm=RUNHEADER.m_max_grad_norm,
        learning_rate=7e-4,
        alpha=0.99,
        epsilon=1e-5,
        lr_schedule="linear",
        verbose=0,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=True,
        write_file=False,
    ):

        super(A2C, self).__init__(
            policy=policy,
            env=env,
            verbose=verbose,
            requires_vec_env=True,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = RUNHEADER.m_ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon: float = epsilon
        if RUNHEADER.cosine_lr:
            lr_schedule = "cosine_annealing"
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.on_validation = RUNHEADER.m_on_validation
        self.gn_alpha = RUNHEADER.gn_alpha

        self.graph = None
        self.sess = None
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.latent_weight = None
        self.latent_weight_numpy = None
        self.latent_weight_pi = None
        self.latent_weight_pi_numpy = None
        self.latent_weight_vn = None
        self.latent_weight_vn_numpy = None
        self.vf_loss = None
        self.pi_coef = None
        self.vf_coef = None
        # self.pg_loss_bias = None
        # self.vf_loss_bias = None
        # self.loss_bias = None  # current loss
        # self.stack_loss_bias = [[1, 1]]
        # self.apply_backprop_gradnorm = None
        self.entropy = None
        self.params = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.dynamic_lr = RUNHEADER.dynamic_lr
        self.dynamic_coe = RUNHEADER.dynamic_coe
        self.summary = None
        self.episode_reward = None

        # custom_variables
        self.day_return = None
        self.cur_lr = None
        self.record_tabular = []
        # self.returns_info_ph = None
        # self.hit_ph = None
        # self.loss_bias_ph = None

        # self.buffer_size = RUNHEADER.m_buffer_size
        # self.replay_ratio = RUNHEADER.m_replay_ratio
        # self.replay_start = RUNHEADER.m_replay_start
        # self.main_replay_start = RUNHEADER.m_main_replay_start

        # fixing
        self.total_example = env.env_method("get_total_episode")[0]
        self.buffer_size = env.env_method("get_buffer_size")[0]
        self.main_replay_start = env.env_method("get_main_replay_start")[0]
        self.total_timesteps = env.env_method("get_total_timesteps")[0]
        self.replay_ratio = RUNHEADER.m_replay_ratio
        self.replay_start = RUNHEADER.m_replay_start

        self.offline_timestamps = -1
        self.weights_load_op = []
        self.weights_load_placeholder = []
        self.epoch_track = 0

        self.model_vf_w = None
        self.model_pi_w = None
        self.model_fc_w = None

        self.tensorboard_log_update = RUNHEADER.m_tensorboard_log_update

        self.write_file = write_file
        self.fp_short_term_memory = "./short_term_memory_log.txt"

        if _init_setup_model:
            self.setup_model()

    def _setup_model_v1(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), (
                "Error: the input policy for the A2C model must be an "
                "instance of common.policies.ActorCriticPolicy."
            )
            self.graph = tf.Graph()
            with self.graph.as_default():
                # self.apply_backprop_gradnorm = tf.constant([0])  # dummy for run

                self.sess = tf_util.make_session(graph=self.graph)
                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, LstmPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    self.n_envs,
                    1,
                    n_batch_step,
                    reuse=False,
                    **self.policy_kwargs,
                )

                with tf.compat.v1.variable_scope(
                    "train_model",
                    reuse=True,
                    custom_getter=tf_util.outer_scope_getter("train_model"),
                ):
                    train_model = self.policy(
                        self.sess,
                        self.observation_space,
                        self.action_space,
                        self.n_envs,
                        self.n_steps,
                        n_batch_train,
                        reuse=True,
                        **self.policy_kwargs,
                    )

                with tf.compat.v1.variable_scope("loss", reuse=False):
                    self.actions_ph = train_model.pdtype.sample_placeholder(
                        [None],
                        name="action_ph",
                    )
                    self.advs_ph = tf.compat.v1.placeholder(
                        tf.float32, [None, RUNHEADER.num_y], name="advs_ph"
                    )
                    self.rewards_ph = tf.compat.v1.placeholder(
                        tf.float32, [None, RUNHEADER.num_y], name="rewards_ph"
                    )
                    self.learning_rate_ph = tf.compat.v1.placeholder(
                        tf.float32, [], name="learning_rate_ph"
                    )
                    # self.loss_bias_ph = tf.compat.v1.placeholder(
                    #     tf.float32, [None, 2], name="loss_bias_ph"
                    # )
                    neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
                    self.entropy = tf.reduce_mean(
                        input_tensor=train_model.proba_distribution.entropy()
                    )
                    # self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                    self.pg_loss = tf.reduce_mean(input_tensor=neglogpac)
                    self.vf_loss = mse(
                        tf.squeeze(train_model.value_fn), self.rewards_ph
                    )
                    self.pi_coef = tf.convert_to_tensor(
                        value=RUNHEADER.m_pi_coef, dtype=tf.float32
                    )
                    self.vf_coef = tf.convert_to_tensor(
                        value=RUNHEADER.m_vf_coef, dtype=tf.float32
                    )

                    self.latent_weight = train_model.latent_weight[-1]
                    self.latent_weight_pi = neglogpac[-1]
                    self.latent_weight_vn = tf.reduce_mean(
                        tf.square(tf.squeeze(train_model.value_fn) - self.rewards_ph),
                        axis=1,
                    )[-1]

                    with ops.control_dependencies([self.pi_coef, self.vf_coef]):
                        barrier_coef = control_flow_ops.no_op(
                            name="update_barrier_coef"
                        )

                    reg_loss_1 = tf.reduce_sum(
                        input_tensor=tf.compat.v1.get_collection(
                            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
                        )
                    )
                    # self.entropy is for the exploration so that disable it on this task
                    loss_1 = self.pg_loss * self.pi_coef - self.entropy * self.ent_coef
                    loss_2 = self.vf_loss * self.vf_coef
                    loss = loss_1 + loss_2 + reg_loss_1
                    # self.loss_bias = tf.stack([[loss_1, loss_2]])

                    if self.full_tensorboard_log:
                        tf.compat.v1.summary.scalar("entropy_loss", self.entropy)
                        tf.compat.v1.summary.scalar(
                            "policy_gradient_loss", self.pg_loss
                        )
                        tf.compat.v1.summary.scalar("value_function_loss", self.vf_loss)
                        tf.compat.v1.summary.scalar(
                            "value_function",
                            tf.reduce_mean(input_tensor=train_model.value_fn),
                        )
                        tf.compat.v1.summary.scalar(
                            "advs", tf.reduce_mean(input_tensor=self.advs_ph)
                        )
                        tf.compat.v1.summary.scalar(
                            "neglogpac", tf.reduce_mean(input_tensor=neglogpac)
                        )
                        tf.compat.v1.summary.scalar("loss", loss)
                        tf.compat.v1.summary.scalar("loss_1", loss_1)
                        tf.compat.v1.summary.scalar("loss_2", loss_2)
                        tf.compat.v1.summary.scalar("reg_loss_1", reg_loss_1)

                #  Make sure                                                                                                                                                                                                                                                                                                                                                                                                     are computed before total_loss.
                update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS, "train_model")
                if update_ops:
                    with ops.control_dependencies(update_ops):
                        barrier = control_flow_ops.no_op(name="update_barrier")
                    new_loss = control_flow_ops.with_dependencies(
                        [barrier, barrier_coef], loss
                    )
                else:
                    new_loss = control_flow_ops.with_dependencies([barrier_coef], loss)

                # calculate gradient
                self.params = find_trainable_variables("model")
                self.ema_params = find_moving_mean_varience(
                    "model"
                )  # just global variables
                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(
                        tf.gradients(ys=new_loss, xs=self.params), self.max_grad_norm
                    )
                else:
                    grads = tf.gradients(ys=new_loss, xs=self.params)
                _grads = list(zip(grads, self.params))
                merge_grads = _grads

                # Todo: check it later
                with tf.compat.v1.variable_scope("weight_visualization", reuse=False):
                    # Todo: check it later
                    if self.full_tensorboard_log:
                        for partial_deviation, param in merge_grads:
                            if partial_deviation is not None:
                                tf.compat.v1.summary.scalar(
                                    "partial_dev_" + param.name,
                                    tf.reduce_mean(input_tensor=partial_deviation),
                                )
                                tf.compat.v1.summary.scalar(
                                    param.name, tf.reduce_mean(input_tensor=param)
                                )
                                tf.compat.v1.summary.histogram(
                                    "partial_dev_" + param.name, partial_deviation
                                )
                                tf.compat.v1.summary.histogram(param.name, param)

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.compat.v1.summary.scalar(
                        "discounted_rewards",
                        tf.reduce_mean(input_tensor=self.rewards_ph),
                    )
                    tf.compat.v1.summary.scalar(
                        "learning_rate",
                        tf.reduce_mean(input_tensor=self.learning_rate_ph),
                    )
                    tf.compat.v1.summary.scalar(
                        "advantage", tf.reduce_mean(input_tensor=self.advs_ph)
                    )

                    if self.full_tensorboard_log:
                        tf.compat.v1.summary.histogram(
                            "discounted_rewards", self.rewards_ph
                        )
                        tf.compat.v1.summary.histogram(
                            "learning_rate", self.learning_rate_ph
                        )
                        tf.compat.v1.summary.histogram("advantage", self.advs_ph)

                        if tf_util.is_image(self.observation_space):
                            tf.compat.v1.summary.image(
                                "processed_observation", train_model.processed_obs
                            )
                        else:
                            tf.compat.v1.summary.histogram(
                                "processed_observation", train_model.processed_obs
                            )
                            tf.compat.v1.summary.scalar(
                                "processed_observation_mean",
                                tf.reduce_mean(input_tensor=train_model.processed_obs),
                            )
                            tf.compat.v1.summary.scalar(
                                "processed_observation_sum",
                                tf.reduce_sum(input_tensor=train_model.processed_obs),
                            )

                with tf.compat.v1.variable_scope("batch_parameter", reuse=False):
                    if self.full_tensorboard_log:
                        _list = ops.get_default_graph().get_collection(
                            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                            "model/Stem_1_1/batch_normalization",
                        )
                        for item in _list:
                            tf.compat.v1.summary.scalar(
                                item.name, tf.reduce_mean(input_tensor=item.value())
                            )
                        _list = ops.get_default_graph().get_collection(
                            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                            "model/Stem_1_2/batch_normalization",
                        )
                        for item in _list:
                            tf.compat.v1.summary.scalar(
                                item.name, tf.reduce_mean(input_tensor=item.value())
                            )

                trainer = tf.compat.v1.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate_ph,
                    decay=self.alpha,
                    epsilon=self.epsilon,
                )
                if update_ops:
                    grad_updates = trainer.apply_gradients(merge_grads)
                    # Ensure the train_tensor computes grad_updates.
                    train_op = control_flow_ops.with_dependencies(
                        [grad_updates], new_loss
                    )
                    train_ops = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                    # Add the operation used for training to the 'train_op' collection
                    if train_op not in train_ops:
                        train_ops.append(train_op)
                    self.apply_backprop = train_op
                else:
                    self.apply_backprop = trainer.apply_gradients(merge_grads)

                # load weights from placeholder
                with tf.compat.v1.variable_scope(
                    "assign_params", reuse=tf.compat.v1.AUTO_REUSE
                ):
                    for tensor in self.ema_params:
                        self.weights_load_placeholder.append(
                            tf.compat.v1.placeholder(
                                dtype=tensor.dtype, shape=tensor.get_shape()
                            )
                        )
                        self.weights_load_op.append(
                            tensor.assign(self.weights_load_placeholder[-1])
                        )

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.compat.v1.global_variables_initializer().run(session=self.sess)

                self.summary = tf.compat.v1.summary.merge_all()

    @funTime("graph building")
    def setup_model(self):
        tf.compat.v1.disable_eager_execution()  # True as default on tensorflow2.0
        if RUNHEADER.grad_norm:
            self._setup_model_v2()
        else:
            self._setup_model_v1()

    @funTime("weight_load_ph")
    def weight_load_ph(self, weights):
        load_op, load_placeholder = [], OrderedDict()
        for _load_op, _load_placeholder, _weight in zip(
            self.weights_load_op, self.weights_load_placeholder, weights
        ):
            load_op.append(_load_op)
            load_placeholder[_load_placeholder] = _weight
        self.sess.run(load_op, load_placeholder)

    # sess run
    def query_network(self, td_map, identifier="", run_options=None, run_metadata=None):
        if identifier == "full_tensorboard_log":
            run_params = [
                self.summary,
                self.pg_loss,
                self.vf_loss,
                self.entropy,
                self.pi_coef,
                self.vf_coef,
                self.apply_backprop,
            ]
        elif identifier == "q_latent_weight":
            run_params = [
                self.pg_loss,
                self.vf_loss,
                self.entropy,
                self.pi_coef,
                self.vf_coef,
                self.latent_weight,
                self.latent_weight_pi,
                self.latent_weight_vn,
                self.apply_backprop,
            ]
        else:
            run_params = [
                self.pg_loss,
                self.vf_loss,
                self.entropy,
                self.pi_coef,
                self.vf_coef,
                self.apply_backprop,
            ]
        return self.sess.run(
            run_params, td_map, options=run_options, run_metadata=run_metadata
        )

    def _train_step(
        self,
        obs,
        states,
        rewards,
        masks,
        actions,
        values,
        update,
        writer=None,
        replay=False,
    ):
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

        def lr_func(predefined_fixed_lr, cosine_lr):
            if cosine_lr:
                return self.learning_rate_schedule.value()
            else:
                return predefined_fixed_lr

        advs = rewards - values
        # explained_var = explained_variance(values, rewards)
        if RUNHEADER.m_online_buffer:
            self.cur_lr = None
            for _ in range(len(obs)):
                self.cur_lr = self.learning_rate_schedule.value()
            assert (
                self.cur_lr is not None
            ), "Error: the observation input array cannon be empty"
        else:  # learning rate for offline learning
            if RUNHEADER.dynamic_lr:
                self.cur_lr = self.learning_rate_schedule.value()
                if self.cur_lr <= RUNHEADER.m_min_learning_rate:
                    self.cur_lr = RUNHEADER.m_min_learning_rate
            else:
                if RUNHEADER.dynamic_coe:
                    assert False, "Op are disabled!!!"

                if RUNHEADER.predefined_fixed_lr is None:
                    self.cur_lr = RUNHEADER.m_offline_learning_rate
                else:
                    self.cur_lr = RUNHEADER.predefined_fixed_lr[0]

        # define td_map
        td_map = {
            self.train_model.obs_ph: obs,
            self.actions_ph: actions,
            self.advs_ph: advs,
            self.rewards_ph: rewards,
            self.learning_rate_ph: self.cur_lr,
        }

        # update td_map
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if writer is not None:  # used in generate buffer
            if (
                self.full_tensorboard_log
                and (1 + update) % self.tensorboard_log_update == 0
            ):
                run_metadata = tf.compat.v1.RunMetadata()
                (
                    summary,
                    policy_loss,
                    value_loss,
                    policy_entropy,
                    pi_coef,
                    vf_coef,
                    _,
                ) = self.query_network(
                    td_map,
                    identifier="full_tensorboard_log",
                    run_options=tf.compat.v1.RunOptions(),
                    run_metadata=run_metadata,
                )
                #  Add for replay memory
                if not replay:
                    writer.add_run_metadata(
                        run_metadata, f"step{update * (self.n_batch + 1):%d}"
                    )
            else:
                (
                    summary,
                    policy_loss,
                    value_loss,
                    policy_entropy,
                    pi_coef,
                    vf_coef,
                    _,
                ) = self.query_network(
                    td_map,
                    identifier="full_tensorboard_log",
                )
            if not replay:
                writer.add_summary(summary, update * (self.n_batch + 1))
        else:  # offline learning
            if RUNHEADER._debug_on and (np.random.randint(1, 200) == 1):
                (
                    policy_loss,
                    value_loss,
                    policy_entropy,
                    pi_coef,
                    vf_coef,
                    latent_weight,
                    latent_weight_pi,
                    latent_weight_vn,
                    _,
                ) = self.query_network(td_map, identifier="q_latent_weight")
                self.latent_weight_numpy = latent_weight
                self.latent_weight_pi_numpy = latent_weight_pi
                self.latent_weight_vn_numpy = latent_weight_vn
            else:
                (
                    policy_loss,
                    value_loss,
                    policy_entropy,
                    pi_coef,
                    vf_coef,
                    _,
                ) = self.query_network(td_map, identifier="general")
                self.latent_weight_numpy = None
                self.latent_weight_pi_numpy = None
                self.latent_weight_vn_numpy = None

        return (
            policy_loss,
            value_loss,
            policy_entropy,
            pi_coef,
            vf_coef,
        )

    def offline_learning_with_buffer(self, runner, writer, model_location):

        print_out_csv, print_out_csv_colname = None, None
        epoch_print_out_csv = []

        file_names = os.listdir(RUNHEADER.m_offline_buffer_file)

        # Merge all the chunks of buffers
        buffer_names = [
            buffer_name
            for buffer_name in file_names
            if ".pkl" and "buffer_" in buffer_name
        ]

        buffer_dict = {
            buffer_names[idx]: RUNHEADER.m_offline_buffer_file
            + "/"
            + buffer_names[idx][:-4]
            for idx in range(len(buffer_names))
        }

        # Learning
        for epoch in range(RUNHEADER.m_offline_learning_epoch):
            for key in buffer_dict.keys():
                learn_data = loadFile(
                    buffer_dict[key]
                )  # learning with chunk by default - dont change it.
                print_out_csv_tmp, print_out_csv_colname = self.offline_learn(
                    writer, runner, learn_data, model_location, epoch
                )
                if print_out_csv is None:
                    print_out_csv = np.array(print_out_csv_tmp)
                else:
                    print_out_csv = np.append(
                        print_out_csv, np.array(print_out_csv_tmp), axis=0
                    )

            aa = np.mean(np.array(print_out_csv), axis=0).tolist()
            aa = aa[:7] + [np.mean(aa[7])]
            epoch_print_out_csv.append(aa)

        pd.DataFrame(epoch_print_out_csv, columns=print_out_csv_colname).to_csv(
            f"{model_location}/record_tabular_epoch_summary.csv"
        )
        # re-init tabular summary for simulation mode
        self.record_tabular = list()
        # break after offline learning
        os._exit(0)

    def learn(
        self,
        total_timesteps,
        callback=None,
        seed=None,
        log_interval=10,
        tb_log_name="A2C",
        reset_num_timesteps=True,
        model_location=None,
    ):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        total_timesteps = (
            self.total_timesteps if total_timesteps is None else total_timesteps
        )

        with SetVerbosity(self.verbose), TensorboardWriter(
            self.graph, self.tensorboard_log, tb_log_name, new_tb_log
        ) as writer:
            self._setup_learn(seed)

            if not RUNHEADER.m_online_buffer:
                learning_timestemp = (
                    self.n_envs
                    * self.n_steps
                    * self.total_example
                    * RUNHEADER.m_offline_learning_epoch
                )
                self.learning_rate_schedule = Scheduler(
                    initial_value=RUNHEADER.m_offline_learning_rate,
                    n_values=learning_timestemp,
                    schedule=self.lr_schedule,
                    cyclic_lr_min=RUNHEADER.cyclic_lr_min,
                    cyclic_lr_max=RUNHEADER.cyclic_lr_max,
                    total_step=int(self.total_example / self.n_envs)
                    * RUNHEADER.m_offline_learning_epoch,
                )
                runner = A2CRunner(
                    self.env, self, n_steps=self.n_steps, gamma=self.gamma
                )
                self.episode_reward = np.zeros((self.n_envs,))

                # load offline-buffer and learning and exit after learning
                writer = None  # force to write is None
                self.offline_learning_with_buffer(runner, writer, model_location)

            else:  # generate buffer
                learning_timestemp = total_timesteps
                self.learning_rate_schedule = Scheduler(
                    initial_value=self.learning_rate,
                    n_values=learning_timestemp,
                    schedule="linear",
                )
                runner = A2CRunner(
                    self.env, self, n_steps=self.n_steps, gamma=self.gamma
                )
                self.episode_reward = np.zeros((self.n_envs,))

                buffer = type_of_buffer(
                    self.env, self.n_steps, self.buffer_size, RUNHEADER.as_chunk
                )
                # simulation learning or continue the simulation learning after the offline learning
                print_rewards = []
                current_timesteps = 0
                current_episode_idx = 0
                ep_dsr = []
                ep_dones = []
                entry_th = None
                mask_th = None
                learned_experiences_ep = 0
                non_pass_episode = 0
                short_term_roll_out = 0
                delete_repeat_sample = 0  # 5times repeat

                for update in range(1, total_timesteps // self.n_batch + 1):
                    if update == self.total_example:
                        os._exit(0)  # without raising SystemExit

                    # 1. set current observation episode
                    if np.sum(self.env.get_attr("eoe")):
                        current_episode_idx = 0
                    self.env.set_attr("current_episode_idx", current_episode_idx)
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

                    # gather
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    short_term_buffer = SBuffer(env=self.env, n_steps=self.n_steps)
                    while bool_runner:
                        self.env.env_method("clear_cache")
                        # step memory
                        (
                            obs,
                            states,
                            rewards,
                            masks,
                            actions,
                            values,
                        ) = runner.run()

                        # short-term memory
                        (
                            bool_runner,
                            entry_th,
                            mask_th,
                            cond_cnt_1,
                            cond_cnt_2,
                            max_iter,
                            exit_cnt,
                            short_term_buffer,
                        ) = self.short_term_simulation(
                            rewards,
                            masks,
                            entry_th,
                            mask_th,
                            cond_cnt_1,
                            cond_cnt_2,
                            max_iter,
                            bool_runner,
                            exit_cnt,
                            short_term_buffer,
                            obs,
                            actions,
                            values,
                            states,
                            current_iter_cnt,
                        )
                        sys.stdout.write(
                            f"\r>> [{current_iter_cnt:}] short-term memory:  {short_term_buffer.num_in_buffer}/{self.n_envs}"
                        )
                        sys.stdout.flush()
                        current_iter_cnt = current_iter_cnt + 1
                    # gather end

                    if short_term_buffer.num_in_buffer > 0:
                        # caution: this is a short-term simulation roll-out not a step-simulation one
                        short_term_roll_out = int(
                            self.n_envs - short_term_buffer.num_in_buffer
                        )
                        """ to calculate mean-variance of experience
                        """
                        # mark 1.
                        (
                            obs,
                            actions,
                            rewards,
                            values,
                            states,
                            masks,
                        ) = short_term_buffer.get()
                    current_episode_idx = current_episode_idx + 1

                    # 3. add gathered experiences to buffer
                    # if buffer is not None:
                    if (buffer is not None) and (short_term_buffer.num_in_buffer > 0):
                        buffer.put(
                            obs,
                            actions,
                            rewards,
                            values,
                            states,
                            masks,
                        )

                    # 4. re-call experiences [sub-process of replay]
                    if (
                        buffer is not None
                        and self.replay_ratio > 0
                        and buffer.has_atleast(self.n_envs)
                    ):
                        if short_term_buffer.num_in_buffer == 0:
                            (
                                obs,
                                actions,
                                rewards,
                                values,
                                states,
                                masks,
                                learned_experiences_ep,
                                non_pass_episode,
                            ) = self.re_call_buff(
                                buffer, learned_experiences_ep, non_pass_episode, runner
                            )

                    # 5. fit training model
                    (
                        policy_loss,
                        value_loss,
                        policy_entropy,
                        pi_coef,
                        vf_coef,
                    ) = self._train_step(
                        obs,
                        states,
                        rewards,
                        masks,
                        actions,
                        values,
                        self.num_timesteps // (self.n_batch + 1),
                        writer,
                    )
                    # self.pg_loss_bias, self.vf_loss_bias = (
                    #     policy_loss,
                    #     value_loss,
                    # )

                    """Blows describe post-processes during the training
                    """
                    # model save according to the time stamps
                    current_timesteps = current_timesteps + 1
                    # drop a model with every 0.5% of examples and buffers
                    # basically sample generation section or online learning + sample generation section
                    # if update % int(self.total_example // 200) == 0:
                    # if update % int(self.total_example * 0.5) == 0:
                    #     model_name = f"{model_location}/fs_{current_timesteps}_ev{np.mean(explained_var):3.3}_pe{policy_entropy:3.3}_pl{policy_loss:3.3}_vl{value_loss:3.3}.pkl"
                    #     self.save(model_name)

                    # buffer save
                    if (buffer.num_in_buffer >= buffer.size) or (
                        update == self.total_example - 1
                    ):
                        buffer.remove_unfilled()
                        writeFile(
                            f"{RUNHEADER.m_offline_buffer_file}/buffer_E{self.n_envs}_S{self.n_steps}_U{update}",
                            buffer,
                        )

                        buffer = None
                        buffer = type_of_buffer(
                            self.env, self.n_steps, self.buffer_size, RUNHEADER.as_chunk
                        )  # re-init

                    # if writer is not None:
                    #     self.episode_reward = total_episode_reward_logger(
                    #         self.episode_reward,
                    #         true_reward.reshape(
                    #             (self.n_envs, self.n_steps * true_reward.shape[-1])
                    #         ),
                    #         masks.reshape((self.n_envs, self.n_steps)),
                    #         writer,
                    #         self.num_timesteps,
                    #     )
                    self.num_timesteps += self.n_batch + 1

                    """Write printout and save csv
                    """
                    explained_var = self.print_out_tabular(
                        update=update,
                        values=values,
                        rewards=rewards,
                        policy_entropy=policy_entropy,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        model_location=model_location,
                        entry_th=entry_th,
                        mask_th=mask_th,
                        buffer=buffer,
                        learned_experiences_ep=learned_experiences_ep,
                        non_pass_episode=non_pass_episode,
                        short_term_roll_out=short_term_roll_out,
                        log_interval=log_interval,
                        pi_coef=pi_coef,
                        vf_coef=vf_coef,
                    )

                    """ Replay [main-process of replay]
                    """
                    if (
                        (self.replay_ratio > 0)
                        and buffer.has_atleast(self.replay_start)
                        and (update > self.main_replay_start)
                    ):
                        samples_number = np.random.poisson(self.replay_ratio)
                        for _ in range(samples_number * replay_iteration):
                            # get obs, actions, rewards, mus, dones from buffer.
                            (
                                obs,
                                states,
                                rewards,
                                masks,
                                actions,
                                values,
                            ) = self.retrive_experiences(buffer, runner)

                            self._train_step(
                                obs,
                                states,
                                rewards,
                                masks,
                                actions,
                                values,
                                self.num_timesteps // (self.n_batch + 1),
                                writer,
                                replay=True,
                            )
                        learned_experiences_ep = learned_experiences_ep + samples_number

        return self

    def retrive_experiences(self, buffer, runner, wrs=None):
        if wrs is None:
            (
                obs_buffer,
                action_buffer,
                rewards_buffer,
                dones_buffer,
            ) = buffer.get_rx()
        else:
            (
                obs_buffer,
                action_buffer,
                rewards_buffer,
                dones_buffer,
            ) = buffer.get_non_replacement(wrs)

        return runner.run(
            obs_buffer=obs_buffer,
            action_buffer=action_buffer,
            rewards_buffer=rewards_buffer,
            dones_buffer=dones_buffer,
            online_buffer=False,
        )

    def offline_learn(self, writer, runner, buffer, model_location, epoch):

        buffer.n_env = (
            self.n_envs
        )  # rewrite - the numbers of generate and train agent might be changed
        samples_number = int(buffer.num_in_buffer / buffer.n_env)
        print_out_csv = []
        rewards, values = None, None
        print_out_csv_colname = None

        without_replacement_sampling = np.arange(buffer.num_in_buffer)
        for print_epoch_result in range(RUNHEADER.m_sub_epoch):
            np.random.shuffle(without_replacement_sampling)

            # for update in range(samples_number):  # learning for 1 epoch.. it investigates samples in a offline file at n_env times
            for update in range(samples_number + 1):  # learning for 1 epoch
                r_idx = np.arange(update * buffer.n_env, (update + 1) * buffer.n_env)
                if r_idx[-1] >= buffer.num_in_buffer:
                    dummy_idx = (r_idx[-1] - buffer.num_in_buffer) + 1
                    r_idx[-dummy_idx:] = np.random.randint(
                        0, buffer.num_in_buffer, dummy_idx
                    )

                wrs = without_replacement_sampling[r_idx]
                (
                    obs,
                    states,
                    rewards,
                    masks,
                    actions,
                    values,
                ) = self.retrive_experiences(buffer, runner, wrs)

                self.offline_timestamps = self.offline_timestamps + 1
                (
                    policy_loss,
                    value_loss,
                    policy_entropy,
                    pi_coef,
                    vf_coef,
                ) = self._train_step(
                    obs,
                    states,
                    rewards,
                    masks,
                    actions,
                    values,
                    self.offline_timestamps,
                    # self.num_timesteps // (self.n_batch + 1),
                    writer,
                    replay=False,
                )
                # self.pg_loss_bias, self.vf_loss_bias = (
                #     policy_loss,
                #     value_loss,
                # )
                explained_var = self.print_out_tabular(
                    policy_entropy=policy_entropy,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    model_location=model_location,
                    rewards=rewards,
                    values=values,
                    opt=1,
                    pi_coef=pi_coef,
                    vf_coef=vf_coef,
                )

                print_out_csv.append(
                    [
                        epoch,
                        update,
                        policy_entropy,
                        policy_loss,
                        value_loss,
                        float(
                            np.mean(
                                np.reshape(
                                    rewards,
                                    [buffer.n_env, self.n_steps, rewards.shape[-1]],
                                )
                            )
                        ),
                        float(
                            np.mean(
                                np.reshape(
                                    values,
                                    [buffer.n_env, self.n_steps, values.shape[-1]],
                                )
                            )
                        ),
                        explained_var,
                    ]
                )
                print_out_csv_colname = [
                    "epoch",
                    "update",
                    "policy_entropy",
                    "policy_loss",
                    "value_loss",
                    "rewards",
                    "values",
                    "explained_var",
                ]

                # sys.stdout.write('\r>> [{}] Offline Learning step: {}/{}'.format(epoch, update, samples_number))
                sys.stdout.write(
                    f"\r>> [{epoch}] Offline Learning step: {(update + 1) * buffer.n_env}/{buffer.num_in_buffer}"
                )
                sys.stdout.flush()

            # otherwise too much model are dropped. configure it according to the experimental result
            if (
                print_epoch_result == (RUNHEADER.m_sub_epoch - 1)
                and self.epoch_track <= epoch
            ):
                self.epoch_track = self.epoch_track + 1
                # model drop
                # Drop the model
                d_time = (
                    str(datetime.datetime.now())[:-10]
                    .replace(":", "-")
                    .replace("-", "")
                    .replace(" ", "_")
                )

                if RUNHEADER.m_train_mode == 0:
                    prefix = ""
                else:
                    prefix = "FT"

                model_name = "{}/{}_sub_epo_{}_pe{:3.3}_pl{:3.3}_vl{:3.3}_ev{:3.3}.pkl".format(
                    model_location,
                    d_time,
                    epoch,
                    np.mean(np.array(print_out_csv)[:, 2]),
                    # policy_entropy
                    np.mean(np.array(print_out_csv)[:, 3]),
                    # policy_loss
                    np.mean(np.array(print_out_csv)[:, 4]),
                    # value_loss
                    np.mean(np.mean(np.array(print_out_csv)[:, 7])),
                )  # explained_var

                if RUNHEADER._debug_on:
                    if (int(epoch) % 5) == 1:
                        self.save(model_name)
                else:
                    if (epoch >= RUNHEADER.c_epoch) and ((int(epoch) % 2) == 1):
                        self.save(model_name)

                # # not in use - memory overflow
                # self.validation_test(runner, self.initial_state, epoch, model_name)

            # if True:  # drops all models corresponding each epochs
            #     # model drop
            #     # Drop the model
            #     d_time = str(datetime.datetime.now())[:-10].replace(':', '-').replace('-', '').replace(' ', '_')
            #     model_name = '{}/{}_sub_epo_{}_pe{:3.3}_pl{:3.3}_vl{:3.3}_ev{:3.3}.pkl'.format(model_location, d_time, epoch,
            #                                                                                     np.mean(policy_entropy_summary),
            #                                                                                     np.mean(policy_loss_summary),
            #                                                                                     np.mean(value_loss_summary), np.mean(explained_var_summary))
            #     self.save(model_name)
            #     self.validation_test(runner, self.initial_state, epoch, model_name)

            # disable it. useless
            # _ = self.print_out_tabular(policy_entropy=np.mean(policy_entropy_summary),
            #                            policy_loss=np.mean(policy_loss_summary),
            #                            value_loss=np.mean(value_loss_summary), model_location=model_location,
            #                            rewards=rewards, values=values,
            #                            opt=1)
        return print_out_csv, print_out_csv_colname

    def re_call_buff(self, buffer, learned_experiences_ep, non_pass_episode, runner):

        (
            obs,
            states,
            rewards,
            masks,
            actions,
            values,
        ) = self.retrive_experiences(buffer, runner)

        learned_experiences_ep = learned_experiences_ep + 1
        non_pass_episode = non_pass_episode + 1

        return (
            obs,
            actions,
            rewards,
            values,
            states,
            masks,
            learned_experiences_ep,
            non_pass_episode,
        )

    # @funTime('save model')
    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            # "vf_coef": self.vf_coef,
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
            "policy_kwargs": self.policy_kwargs,
        }

        # params = self.sess.run(self.params)  # params are only used for training and its results shared with global parameters
        ema_params = self.sess.run(
            self.ema_params
        )  # save ema_params updated 2020-02-18

        self._save_to_file(save_path, data=data, params=ema_params)

    def _short_term_simulation(self, rewards, masks, actions):
        # dones
        cond = np.reshape(masks, [self.n_envs, self.n_steps])
        cond_idx = np.squeeze(
            np.argwhere(
                np.array([util.consecutive_true(item) for item in cond])
                > (self.n_steps - 1)
            ),
            axis=1,
        ).tolist()

        env_idx = list()
        for idx in cond_idx:
            env_idx.append(idx)
        return env_idx

    def short_term_simulation(
        self,
        rewards,
        masks,
        entry_th,
        mask_th,
        cond_cnt_1,
        cond_cnt_2,
        max_iter,
        bool_runner,
        exit_cnt,
        short_term_buffer,
        obs,
        actions,
        values,
        states,
        current_iter_cnt,
    ):

        #  store examples to short-term memory
        env_idx = self._short_term_simulation(rewards, masks, actions)
        short_term_buffer.put(
            obs,
            actions,
            rewards,
            values,
            states,
            masks,
            env_idx,
        )
        max_iter = max_iter + 1

        # exit condition
        if exit_cnt == RUNHEADER.m_exit_cnt:
            bool_runner = False
        if short_term_buffer.num_in_buffer == self.n_envs:
            bool_runner = False
        # early-stop for short-term memory
        if short_term_buffer.num_in_buffer > RUNHEADER.m_early_stop:
            bool_runner = False

        return (
            bool_runner,
            entry_th,
            mask_th,
            cond_cnt_1,
            cond_cnt_2,
            max_iter,
            exit_cnt,
            short_term_buffer,
        )

    def print_out_tabular(
        self,
        update=None,
        values=None,
        rewards=None,
        policy_entropy=None,
        policy_loss=None,
        value_loss=None,
        model_location=None,
        opt=0,
        entry_th=None,
        mask_th=None,
        buffer=None,
        learned_experiences_ep=None,
        non_pass_episode=None,
        short_term_roll_out=None,
        log_interval=None,
        pi_coef=None,
        vf_coef=None,
    ):
        """Write printout and save csv"""
        if opt == 0:  # default tabular
            explained_var = explained_variance(values, rewards)
            if RUNHEADER._debug_on:
                tabular_column_name = [
                    "n_updates",
                    "total_timesteps",
                    "learning_rate",
                    "policy_entropy",
                    "policy_loss",
                    "value_loss",
                    "explained_variance",
                    "Episode_Rewards",
                    "values",
                    "current_buff_size",
                    "learned_replay_ep",
                    "non_pass_episode",
                    "date",
                    "short_term_roll_out",
                ]
                self.record_tabular.append(
                    [
                        str(update) + "/" + str(self.total_example),
                        self.num_timesteps,
                        self.cur_lr,
                        float(policy_entropy),
                        float(policy_loss),
                        float(value_loss),
                        float(explained_var),
                        float(
                            np.mean(
                                np.reshape(
                                    rewards,
                                    [self.n_envs, self.n_steps, rewards.shape[-1]],
                                )
                            )
                        ),
                        float(
                            np.mean(
                                np.reshape(
                                    values,
                                    [self.n_envs, self.n_steps, values.shape[-1]],
                                )
                            )
                        ),
                        int(buffer.num_in_buffer),
                        learned_experiences_ep,
                        non_pass_episode,
                        g_date,
                        short_term_roll_out,
                    ]
                )
                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    print("\n")
                    print_out_tabular = self.record_tabular[-1]
                    for idx in range(len(tabular_column_name)):
                        logger.record_tabular(
                            tabular_column_name[idx], print_out_tabular[idx]
                        )
                    logger.dump_tabular()
                pd.DataFrame(
                    data=np.array(self.record_tabular), columns=tabular_column_name
                ).to_csv("{}/record_tabular.csv".format(model_location))
        else:  # tabular print for offline learning
            explained_var = explained_variance(values, rewards)
            if RUNHEADER._debug_on:
                tabular_column_name = [
                    "learning_rate",
                    "policy_entropy",
                    "policy_loss",
                    "value_loss",
                    "explained_variance",
                    "loss",
                    "values",
                    "rewards",
                    "advs",
                    "policy_loss_with_coef",
                    "value_loss_with_coef",
                ]
                values = float(
                    np.mean(
                        np.reshape(
                            values, [self.n_envs, self.n_steps, values.shape[-1]]
                        )
                    )
                )
                rewards = float(
                    np.mean(
                        np.reshape(
                            rewards, [self.n_envs, self.n_steps, rewards.shape[-1]]
                        )
                    )
                )
                self.record_tabular.append(
                    [
                        self.cur_lr,
                        float(policy_entropy),
                        float(policy_loss),
                        float(value_loss),
                        float(explained_var),
                        (
                            policy_loss * pi_coef
                            - policy_entropy * RUNHEADER.m_ent_coef
                            + value_loss * vf_coef
                        ),
                        float(values),
                        float(rewards),
                        float(rewards - values),
                        float(policy_loss * pi_coef),
                        float(value_loss * vf_coef),
                    ]
                )
                if self.verbose >= 1:
                    print("\n")
                    print_out_tabular = self.record_tabular[-1]
                    for idx, it in enumerate(tabular_column_name):
                        logger.record_tabular(it, print_out_tabular[idx])
                    logger.dump_tabular()
                pd.DataFrame(
                    data=np.array(self.record_tabular), columns=tabular_column_name
                ).to_csv(f"{model_location}/record_tabular_buffer.csv")

                # when weights has been updated, save as a npy file
                if self.latent_weight_numpy is not None:
                    with open(
                        f"{model_location}/latent_weight_pi_vn_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.npy",
                        "wb",
                    ) as fp:
                        np.save(fp, self.latent_weight_numpy)
                        np.save(fp, self.latent_weight_pi_numpy)
                        np.save(fp, self.latent_weight_vn_numpy)
                        fp.close()

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
        self.n_env = RUNHEADER.m_n_cpu
        if isinstance(env.action_space, Discrete):
            self.n_act = env.action_space.n
        elif isinstance(env.action_space, MultiDiscrete):
            self.n_act = env.action_space.shape[-1]
        else:
            self.n_act = env.action_space.shape[-1]
        self.n_batch = n_env * n_steps

    def run(
        self,
        obs_buffer=None,
        action_buffer=None,
        rewards_buffer=None,
        dones_buffer=None,
        online_buffer=True,
    ):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        print("\n")
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []

        # Todo: assume that each episodes are independent (it depends on your perspective to given problem)
        self.states = self.init_state
        mb_states = self.states
        self.dones = [False for _ in range(self.n_env)]

        # step simulation
        for n_steps_idx in range(self.n_steps):
            if online_buffer:  # on-line simulation
                step_memory = self.step_simulation(n_steps_idx=n_steps_idx)
                actions, values, states, obs, rewards, dones, info = step_memory.get()
            else:  # offline-learning or re-call experiences for online simulation
                (
                    actions,
                    values,
                    states,
                    obs,
                    rewards,
                    dones,
                ) = self.step_simulation_offline(
                    n_steps_idx=n_steps_idx,
                    obs_buffer=obs_buffer,
                    action_buffer=action_buffer,
                    rewards_buffer=rewards_buffer,
                    dones_buffer=dones_buffer,
                )
            assert not np.sum(dones), "True condition detected!!!"

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            self.states = states
            self.dones = dones
            mb_rewards.append(rewards)

        # stack step experiences
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = (
            np.asarray(mb_obs, dtype=self.obs.dtype)
            .swapaxes(1, 0)
            .reshape(self.batch_ob_shape)
        )
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()

        if online_buffer:
            # discount/bootstrap off value fn.. backward continuous reward sum
            for n, (rewards, dones, _) in enumerate(
                zip(mb_rewards, mb_dones, last_values)
            ):
                rewards = rewards.tolist()
                mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_env * n_steps, ...], 2D to 1D
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])

        return (
            mb_obs,
            mb_states,
            mb_rewards,
            mb_masks,
            mb_actions,
            mb_values,
        )

    def step_simulation(self, clipped_actions=0, n_steps_idx=None):
        search_cnt = 0
        terminate_cnt = 0
        min_reward = 0
        b_step = True
        mode = False
        disable_dones = False
        sample_th = RUNHEADER.m_sample_th
        step_memory = StepBuffer(env=self.env)
        self.env.set_attr("current_step", n_steps_idx)
        # if n_steps_idx == 1:
        #     print('test')
        self.obs = np.array(self.env.env_method("obs_from_env"))
        while b_step:
            if mode:
                sample_th = np.inf

            """get step example and stack to step memory
            """
            actions, values, states, neglogp = self.model.step(
                self.obs, self.states, self.dones, sample_th=sample_th
            )

            # step actions - for display
            diff_action = np.sum(
                np.where((actions - clipped_actions) == 0, 0, 1), axis=1
            )
            clipped_actions = actions
            num_clipped_actions = np.sum(clipped_actions, axis=1)
            change_rate = float(np.mean(diff_action / (num_clipped_actions + 1e-15)))
            changes = float(np.mean(diff_action))

            # clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.env.action_space.low, self.env.action_space.high
                )

            # get rewards, dones, info and next observation
            obs, rewards, dones, info = self.env.step(clipped_actions)

            assert np.allclose(
                self.obs, obs
            ), "Delete this code later, just for code test!!!"

            # stack an experience
            step_memory.put(
                actions,
                values,
                states,
                obs,
                rewards,
                dones,
                info,
                mode=mode,
                min_reward=min_reward,
                disable_dones=disable_dones,
            )

            """get sample_th and mode corresponding its search iteration 
            """
            if step_memory.num_in_buffer == self.n_env:
                b_step = False
            else:
                sample_th, terminate_cnt, search_cnt = self.get_sample_th(
                    search_cnt, terminate_cnt, step_memory
                )

            num_in_buffer = step_memory.num_in_buffer
            sys.stdout.write(
                "\r>> [%s][%s][%d-%d] step memory:  %d, %d/%d, %3.2f(%3.2f), s_th:%3.2f"
                % (
                    info[0]["date"],
                    terminate_cnt,
                    n_steps_idx,
                    int(np.mean(np.sum(actions, axis=1))),
                    search_cnt,
                    num_in_buffer,
                    self.n_env,
                    int(changes),
                    change_rate,
                    sample_th,
                )
            )
            sys.stdout.flush()

            # delete later
            global g_date
            g_date = info[0]["date"]
            # g_date2 = sample_th

        return step_memory

    def step_simulation_offline(
        self,
        n_steps_idx=None,
        obs_buffer=None,
        action_buffer=None,
        rewards_buffer=None,
        dones_buffer=None,
    ):
        sample_th = RUNHEADER.m_sample_th
        """get step example and stack to step memory
        """
        self.obs = obs_buffer[n_steps_idx]
        _, values, states, _ = self.model.step(
            self.obs, self.states, self.dones, sample_th=sample_th
        )
        actions = action_buffer[n_steps_idx]

        # get rewards, dones, info and next observation
        obs, rewards, dones = (
            self.obs,
            rewards_buffer[n_steps_idx],
            dones_buffer[n_steps_idx],
        )

        return actions, values, states, obs, rewards, dones

    def step_validation(
        self, observation, initial_state, state, mask, deterministic=True
    ):
        if state is None:
            state = initial_state
        if mask is None:
            mask = [False for _ in range(self.n_env)]
        observation = np.array(observation)

        return self.model.step(observation, state, mask, deterministic=deterministic)

    def get_sample_th(self, search_cnt, terminate_cnt, step_memory):
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
                    if (search_cnt > RUNHEADER.m_interval[idx - 1]) and (
                        search_cnt < RUNHEADER.m_interval[idx]
                    ):
                        sample_th = RUNHEADER.m_interval_value[idx - 1]

        """Control parameter setting according to the sample_th
        """
        if sample_th == -np.inf:  # init for re-iteration
            sample_th = RUNHEADER.m_sample_th
            terminate_cnt = terminate_cnt + 1
            if step_memory.num_in_buffer >= RUNHEADER.m_early_stop:
                terminate_cnt = np.inf

        search_cnt = search_cnt + 1
        return sample_th, terminate_cnt, search_cnt
