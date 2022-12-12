import warnings
from abc import ABC
from itertools import zip_longest

import numpy as np
import tensorflow as tf
import tf_slim as slim
from gym.spaces import Discrete, MultiDiscrete

import header.index_forecasting.RUNHEADER as RUNHEADER
from custom_model.index_forecasting.common.input import observation_input
from custom_model.index_forecasting.common.utils import (
    batch_to_seq,
    linear,
    lstm,
    seq_to_batch,
)
from custom_model.index_forecasting.policies.distributions_v1 import (
    BernoulliProbabilityDistribution,
    CategoricalProbabilityDistribution,
    DiagGaussianProbabilityDistribution,
    MultiCategoricalProbabilityDistribution,
    make_proba_dist_type,
)
from custom_model.index_forecasting.policies.net_factory import net_factory

default_cnn = net_factory()


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = (
        []
    )  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = (
        []
    )  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(
                linear(
                    latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)
                )
            )
        else:
            assert isinstance(
                layer, dict
            ), "Error: the net_arch list can only contain ints and dicts"
            if "pi" in layer:
                assert isinstance(
                    layer["pi"], list
                ), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer["pi"]

            if "vf" in layer:
                assert isinstance(
                    layer["vf"], list
                ), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer["vf"]
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(
        zip_longest(policy_only_layers, value_only_layers)
    ):
        if pi_layer_size is not None:
            assert isinstance(
                pi_layer_size, int
            ), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(
                linear(
                    latent_policy,
                    "pi_fc{}".format(idx),
                    pi_layer_size,
                    init_scale=np.sqrt(2),
                )
            )

        if vf_layer_size is not None:
            assert isinstance(
                vf_layer_size, int
            ), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(
                linear(
                    latent_value,
                    "vf_fc{}".format(idx),
                    vf_layer_size,
                    init_scale=np.sqrt(2),
                )
            )

    return latent_policy, latent_value


class BasePolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        reuse=False,
        scale=False,
        obs_phs=None,
        add_action_ph=False,
    ):
        self.n_env = n_env
        self.n_steps = n_steps
        with tf.compat.v1.variable_scope("input", reuse=False):
            if obs_phs is None:
                self.obs_ph, self.processed_obs = observation_input(
                    ob_space, n_batch, scale=scale
                )
                # self.obs_ph, self.processed_obs = observation_input(
                #     ob_space, None, scale=scale
                # )
            else:
                self.obs_ph, self.processed_obs = obs_phs

            self.action_ph = None
            if add_action_ph:
                self.action_ph = tf.compat.v1.placeholder(
                    dtype=ac_space.dtype,
                    shape=(None,) + ac_space.shape,
                    name="action_ph",
                )
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitely (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitely)
        if feature_extraction == "mlp" and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, /
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        reuse=False,
        scale=False,
    ):
        super(ActorCriticPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale
        )
        self.pdtype = make_proba_dist_type(ac_space)
        self.is_discrete = isinstance(ac_space, Discrete)
        self.is_multidiscrete = isinstance(ac_space, MultiDiscrete)
        self.policy = None
        self.proba_distribution = None
        self.value_fn = None
        self.latent_weight = None
        self.deterministic_action = None
        self.initial_state = None
        self.sample_th_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(), name="sample_th_ph"
        )
        # self.offline_actions_ph = tf.placeholder(dtype=tf.int32, shape=(n_env, self.ac_space.nvec.shape[0]),
        #                                          name="offline_actions_ph")

    def _setup_init(self):
        """
        sets up the distibutions, actions, and value
        """
        with tf.compat.v1.variable_scope("output", reuse=True):
            assert (
                self.policy is not None
                and self.proba_distribution is not None
                and self.value_fn is not None
            )
            self.action = self.proba_distribution.sample(sample_th=self.sample_th_ph)
            self.deterministic_action = self.proba_distribution.mode()
            self.neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self.policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(
                self.proba_distribution, DiagGaussianProbabilityDistribution
            ):
                self.policy_proba = [
                    self.proba_distribution.mean,
                    self.proba_distribution.std,
                ]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self.policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(
                self.proba_distribution, MultiCategoricalProbabilityDistribution
            ):
                self.policy_proba = [
                    tf.nn.softmax(categorical.flatparam())
                    for categorical in self.proba_distribution.categoricals
                ]
            else:
                self.policy_proba = []  # it will return nothing
            # self._value = self.value_fn[:, 0]
            self._value = self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError

    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class LstmPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        n_lstm=256,
        reuse=False,
        layers=None,
        net_arch=None,
        act_fun=tf.tanh,
        cnn_extractor=default_cnn,
        layer_norm=False,
        feature_extraction="cnn",
        is_training=False,
        **kwargs,
    ):
        super(LstmPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse,
            scale=(feature_extraction == "cnn"),
        )

        self._kwargs_check(feature_extraction, kwargs)

        with tf.compat.v1.variable_scope("input", reuse=True):
            self.masks_ph = tf.compat.v1.placeholder(
                tf.float32, [n_batch], name="masks_ph"
            )  # mask (done t-1)
            self.states_ph = tf.compat.v1.placeholder(
                tf.float32, [self.n_env, n_lstm * 2], name="states_ph"
            )  # states
            # self.masks_ph = tf.compat.v1.placeholder(
            #     tf.float32, [None], name="masks_ph"
            # )  # mask (done t-1)
            # self.states_ph = tf.compat.v1.placeholder(
            #     tf.float32, [None, n_lstm * 2], name="states_ph"
            # )  # states

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn(
                    "The layers parameter is deprecated. Use the net_arch parameter instead."
                )

            with tf.compat.v1.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(
                        self.processed_obs, is_training, **kwargs
                    )
                else:
                    extracted_features = tf.compat.v1.layers.flatten(self.processed_obs)
                    if RUNHEADER.enable_lstm:
                        for i, layer_size in enumerate(layers):
                            extracted_features = act_fun(
                                linear(
                                    extracted_features,
                                    "pi_fc" + str(i),
                                    n_hidden=layer_size,
                                    init_scale=np.sqrt(2),
                                )
                            )
                if RUNHEADER.enable_lstm:
                    input_sequence = batch_to_seq(
                        extracted_features, self.n_env, n_steps
                    )
                    masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
                    rnn_output, self.snew = lstm(
                        input_sequence,
                        masks,
                        self.states_ph,
                        "lstm1",
                        n_hidden=n_lstm,
                        layer_norm=layer_norm,
                    )
                    rnn_output = seq_to_batch(rnn_output)
                else:
                    self.snew = tf.zeros([self.n_env, n_lstm * 2], tf.float32)
                    rnn_output = extracted_features

                self.latent_weight = rnn_output
                if not RUNHEADER.enable_non_shared_part:
                    value_fn = linear(rnn_output, "vf", RUNHEADER.mtl_target)

                    (
                        self.proba_distribution,
                        self.policy,
                        self.q_value,
                    ) = self.pdtype.proba_distribution_from_latent(
                        rnn_output, rnn_output
                    )
                else:
                    value_latent = linear(rnn_output, "vf_latent", 256)
                    policy_latent = linear(rnn_output, "pi_latent", 256)

                    value_fn = linear(value_latent, "vf", RUNHEADER.mtl_target)

                    (
                        self.proba_distribution,
                        self.policy,
                        self.q_value,
                    ) = self.pdtype.proba_distribution_from_latent(
                        policy_latent, policy_latent
                    )

                    # alternative version - not implemented
                    # # Create non share representations(value and policy) from the shared representation  version 1
                    # latent_layer = None
                    # for k in list(RUNHEADER.mkname_mkidx.keys()):
                    #     if k != "TOTAL":
                    #         latent_layer = slim.layer_norm(
                    #             linear(
                    #                 rnn_output,
                    #                 f"{k}_latent",
                    #                 rnn_output.shape[-1],
                    #                 init_scale=np.sqrt(2),
                    #             ),
                    #             scale=False,
                    #             activation_fn=act_fun,
                    #         )
                    #     if len(latent_layer) == 0:
                    #         concat_layer = latent_layer
                    #     else:
                    #         concat_layer = tf.concat(
                    #             values=[concat_layer, latent_layer], axis=0
                    #         )

                    # # Todo: later on fix - tf.masking 응용 고려
                    # # target market to learn
                    # latent_layer_dim = latent_layer.shape[-1]
                    # learn_target = tf.zeros([concat_layer.shape[-1]])

                    # # Todo: later on fix - tf.masking 응용 고려
                    # self.learn_target_ph = None

                    # if self.learn_target_ph == 0:
                    #     learn_target[:latent_layer_dim] = 1
                    # else:
                    #     learn_target[
                    #         latent_layer_dim
                    #         * self.learn_target_ph : latent_layer_dim
                    #         * (self.learn_target_ph + 1)
                    #     ] = 1

                    # concat_layer = tf.multiply(concat_layer, learn_target)

                    # # function
                    # value_fn = linear(concat_layer, "vf", RUNHEADER.mtl_target)

                    # (
                    #     self.proba_distribution,
                    #     self.policy,
                    #     self.q_value,
                    # ) = self.pdtype.proba_distribution_from_latent(
                    #     concat_layer, concat_layer
                    # )

            self.value_fn = value_fn

        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn(
                    "The new net_arch parameter overrides the deprecated layers parameter."
                )
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.compat.v1.variable_scope("model", reuse=reuse):
                latent = tf.compat.v1.layers.flatten(self.processed_obs)
                policy_only_layers = (
                    []
                )  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = (
                    []
                )  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(
                            linear(
                                latent,
                                "shared_fc{}".format(idx),
                                layer_size,
                                init_scale=np.sqrt(2),
                            )
                        )
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError(
                                "The net_arch parameter must only contain one occurrence of 'lstm'!"
                            )
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(
                            input_sequence,
                            masks,
                            self.states_ph,
                            "lstm1",
                            n_hidden=n_lstm,
                            layer_norm=layer_norm,
                        )
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(
                            layer, dict
                        ), "Error: the net_arch list can only contain ints and dicts"
                        if "pi" in layer:
                            assert isinstance(
                                layer["pi"], list
                            ), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer["pi"]

                        if "vf" in layer:
                            assert isinstance(
                                layer["vf"], list
                            ), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer["vf"]
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError(
                            "LSTMs are only supported in the shared part of the policy network."
                        )
                    assert isinstance(
                        pi_layer_size, int
                    ), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(
                            latent_policy,
                            "pi_fc{}".format(idx),
                            pi_layer_size,
                            init_scale=np.sqrt(2),
                        )
                    )

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError(
                            "LSTMs are only supported in the shared part of the value function "
                            "network."
                        )
                    assert isinstance(
                        vf_layer_size, int
                    ), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(
                            latent_value,
                            "vf_fc{}".format(idx),
                            vf_layer_size,
                            init_scale=np.sqrt(2),
                        )
                    )

                if not lstm_layer_constructed:
                    raise ValueError(
                        "The net_arch parameter must contain at least one occurrence of 'lstm'!"
                    )

                self.value_fn = linear(latent_value, "vf", RUNHEADER.mtl_target)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                (
                    self.proba_distribution,
                    self.policy,
                    self.q_value,
                ) = self.pdtype.proba_distribution_from_latent(
                    latent_policy, latent_value
                )
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, sample_th=0.8):
        # print('is_first_step: {}'.format(is_first_step))
        if deterministic:
            action, _value, snew, neglogp = self.sess.run(
                [
                    self.deterministic_action,
                    self._value,
                    self.snew,
                    self.neglogp,
                ],
                {
                    self.obs_ph: obs,
                    self.states_ph: state,
                    self.masks_ph: mask,
                    self.sample_th_ph: sample_th,
                },
            )
        else:
            action, _value, snew, neglogp = self.sess.run(
                [self.action, self._value, self.snew, self.neglogp],
                {
                    self.obs_ph: obs,
                    self.states_ph: state,
                    self.masks_ph: mask,
                    self.sample_th_ph: sample_th,
                },
            )
        # self.sess.run({self.first_step_of_episode_ph: is_first_step})
        return action, _value, snew, neglogp

    # def rand_action(self, rows, cols, minval=3, maxval=100):
    #     actions = np.zeros(shape=(rows, cols), dtype=np.int)
    #
    #     num_action = np.int(np.random.uniform(low=minval, high=maxval))
    #     for idx in range(rows):
    #         true_idx = np.random.permutation(np.arange(cols))
    #         for k in range(num_action):
    #             actions[idx, true_idx[k]] = 1
    #     return actions

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(
            self.policy_proba,
            {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask},
        )

    def value(self, obs, state=None, mask=None):
        return self.sess.run(
            self._value, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask}
        )


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        reuse=False,
        layers=None,
        net_arch=None,
        act_fun=tf.tanh,
        cnn_extractor=default_cnn,
        feature_extraction="cnn",
        **kwargs,
    ):
        super(FeedForwardPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse=reuse,
            scale=(feature_extraction == "cnn"),
        )

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn(
                "Usage of the `layers` parameter is deprecated! Use net_arch instead "
                "(it has a different semantics though).",
                DeprecationWarning,
            )
            if net_arch is not None:
                warnings.warn(
                    "The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                    DeprecationWarning,
                )

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.compat.v1.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(
                    tf.compat.v1.layers.flatten(self.processed_obs), net_arch, act_fun
                )

            self.value_fn = linear(vf_latent, "vf", RUNHEADER.mtl_target)

            (
                self.proba_distribution,
                self.policy,
                self.q_value,
            ) = self.pdtype.proba_distribution_from_latent(
                pi_latent, vf_latent, init_scale=0.01
            )

        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self._value, self.neglogp],
                {self.obs_ph: obs},
            )
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self._value, self.neglogp], {self.obs_ph: obs}
            )
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs
    ):
        super(CnnPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse,
            feature_extraction="cnn",
            **_kwargs,
        )


class CnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    # default n_lstm=256
    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        n_lstm=256,
        reuse=False,
        **_kwargs,
    ):
        super(CnnLstmPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm,
            reuse,
            layer_norm=False,
            feature_extraction="cnn",
            **_kwargs,
        )


class CnnLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        n_lstm=256,
        reuse=False,
        **_kwargs,
    ):
        super(CnnLnLstmPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm,
            reuse,
            layer_norm=True,
            feature_extraction="cnn",
            **_kwargs,
        )


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs
    ):
        super(MlpPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse,
            feature_extraction="mlp",
            **_kwargs,
        )


class MlpLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        n_lstm=256,
        reuse=False,
        **_kwargs,
    ):
        super(MlpLstmPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm,
            reuse,
            layer_norm=False,
            feature_extraction="mlp",
            **_kwargs,
        )


class MlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        n_lstm=256,
        reuse=False,
        **_kwargs,
    ):
        super(MlpLnLstmPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm,
            reuse,
            layer_norm=True,
            feature_extraction="mlp",
            **_kwargs,
        )


_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
    }
}


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError(
            "Error: the policy type {} is not registered!".format(base_policy_type)
        )
    if name not in _policy_registry[base_policy_type]:
        raise ValueError(
            "Error: unknown policy type {}, the only registed policy type are: {}!".format(
                name, list(_policy_registry[base_policy_type].keys())
            )
        )
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError(
            "Error: the policy {} is not of any known subclasses of BasePolicy!".format(
                policy
            )
        )

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError(
            "Error: the name {} is alreay registered for a different policy, will not override.".format(
                name
            )
        )
    _policy_registry[sub_class][name] = policy
