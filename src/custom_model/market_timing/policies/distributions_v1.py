import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from gym import spaces

from custom_model.market_timing.common.utils import linear
import header.market_timing.RUNHEADER as RUNHEADER

class ProbabilityDistribution(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilites
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability

        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leiber divergence from the given probabilty distribution

        :param other: ([float]) the distibution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probabilty distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """

    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        Returns the probability distribution from flat probabilities
        flat: flattened vector of parameters of probability distribution

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param pi_latent_vector: ([float]) the latent pi values
        :param vf_latent_vector: ([float]) the latent vf values
        :param init_scale: (float) the inital scale of the distribution
        :param init_bias: (float) the inital bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        raise NotImplementedError

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.compat.v1.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.compat.v1.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class CategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_vec):
        """
        The probability distribution type for multiple categorical input

        :param n_vec: ([int]) the vectors
        """
        # Cast the variable because tf does not allow uint32
        self.n_vec = n_vec.astype(np.int32)
        # Check that the cast was valid
        assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"

    def probability_distribution_class(self):
        return MultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiCategoricalProbabilityDistribution(self.n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [sum(self.n_vec)]

    def sample_shape(self):
        return [len(self.n_vec)]

    def sample_dtype(self):
        return tf.int32


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.compat.v1.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.compat.v1.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for bernoulli input

        :param size: (int) the number of dimensions of the bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return BernoulliProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(input=self.logits, axis=-1)

    def neglogp(self, x):
        # return tf.nn. (logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=tf.stop_gradient(one_hot_actions))

    def kl(self, other):
        a_0 = self.logits - tf.reduce_max(input_tensor=self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(input_tensor=other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(input_tensor=exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(input_tensor=exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(input_tensor=p_0 * (a_0 - tf.math.log(z_0) - a_1 + tf.math.log(z_1)), axis=-1)

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(input_tensor=self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(input_tensor=exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(input_tensor=p_0 * (tf.math.log(z_0) - a_0), axis=-1)

    def sample(self, ratio=None, sample_th=np.inf):
        target_dist = tf.zeros(tf.shape(input=self.logits), dtype=self.logits.dtype) + (1 - ratio, ratio)
        uniform = tf.random.uniform(tf.shape(input=self.logits), dtype=self.logits.dtype)
        dist = (target_dist + uniform)/2

        cond = tf.squeeze(tf.random.uniform(shape=(1,)))
        # th = tf.squeeze(tf.constant([0.4]))

        distribution_sample = lambda: tf.argmax(input=self.logits, axis=-1)
        random_samples = lambda: tf.argmax(input=self.logits - tf.math.log(-tf.math.log(dist)), axis=-1)

        samples = tf.case([(tf.greater(cond, sample_th), random_samples)], default=distribution_sample)

        return samples

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))
        self.num_env = self.flat.shape[0]

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    # #  original
    # def sample_original(self):
    #     return tf.cast(tf.stack([p.sample_origin() for p in self.categoricals], axis=-1), tf.int32)

    # def sample(self, first_step_episode=None):
    #     actions_from_uniform = lambda: self._sample(first_step_episode)
    #     default_fn = lambda: self.sample_original()
    #
    #     test = tf.case([(tf.equal(tf.reduce_sum(self._sample(first_step_episode)), 0), actions_from_uniform)],
    #                    default=default_fn)
    #     return test

    def sample(self, sample_th=np.inf):

        target_num_action = tf.random.uniform(shape=(),
                                              minval=RUNHEADER.m_allow_actions_min,
                                              maxval=RUNHEADER.m_allow_actions_max)
        total_action = len(self.categoricals)
        ratio = target_num_action / total_action

        random_samples_free = lambda: \
            tf.stack([tf.random.categorical(logits=tf.math.log([[1 - ratio, ratio]]),
                                            num_samples=len(self.categoricals))[0] for _ in range(self.num_env)])
        actions_from_uniform = lambda: \
            tf.stack([p.sample(ratio, sample_th) for p in self.categoricals], axis=-1)
        # actions_from_uniform = tf.stack([p.sample(ratio, sample_th) for p in self.categoricals], axis=-1)

        samples = tf.case([(tf.equal(sample_th, np.inf), random_samples_free)], default=actions_from_uniform)

        return tf.cast(samples, tf.int32)


    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(input_tensor=tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(input=x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(input_tensor=self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(input_tensor=other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(input_tensor=self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Bounds are taken into acount outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mean + self.std * tf.random.normal(tf.shape(input=self.mean), dtype=self.mean.dtype)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate gaussian input data
        """
        return cls(flat)


class BernoulliProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from bernoulli input

        :param logits: ([float]) the bernoulli input data
        """
        self.logits = logits
        self.probabilities = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.probabilities)

    def neglogp(self, x):
        return tf.reduce_sum(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(x, dtype=tf.float32)),
                             axis=-1)

    def kl(self, other):
        return tf.reduce_sum(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits,
                                                                     labels=self.probabilities), axis=-1) - \
               tf.reduce_sum(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def entropy(self):
        return tf.reduce_sum(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def sample(self):
        samples_from_uniform = tf.random.uniform(tf.shape(input=self.probabilities))
        return tf.cast(math_ops.less(samples_from_uniform, self.probabilities), dtype=tf.float32)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new bernoulli input

        :param flat: ([float]) the bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given bernoulli input data
        """
        return cls(flat)


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the approriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")


def shape_el(tensor, index):
    """
    get the shape of a TensorFlow Tensor element

    :param tensor: (TensorFlow Tensor) the input tensor
    :param index: (int) the element
    :return: ([int]) the shape
    """
    maybe = tensor.get_shape()[index]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(input=tensor)[index]
