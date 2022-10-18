from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from custom_model.index_forecasting.common.utils import conv, linear, conv_to_fc
import header.index_forecasting.RUNHEADER as RUNHEADER
# slim = tf_slim  # call keras
import tf_slim as slim  # call tf.layers


def block5(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 5x10x320 resnet block."""
    with tf.compat.v1.variable_scope(scope, 'block5', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3].value, 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up

        if activation_fn:
            net = activation_fn(net)
    return net


def block3(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 5x5x320 resnet block."""
    with tf.compat.v1.variable_scope(scope, 'block3', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3].value, 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up

        if activation_fn:
            net = activation_fn(net)
    return net


def inception_resnet_v2(scaled_images, is_training=False, **kwargs):
    # on - batch normalization, off - regularization
    with slim.arg_scope(inception_arg_scope(use_batch_norm=False)):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=None,
                            biases_regularizer=None):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net = slim.flatten(scaled_images)
                net = slim.fully_connected(net, int(512 * RUNHEADER.m_num_features), activation_fn=None,
                                           normalizer_fn=None,
                                           scope='Feature_out')

    return net


def inception_arg_scope(weight_decay=RUNHEADER.m_l2_norm,
                        use_batch_norm=True,
                        batch_norm_decay=RUNHEADER.m_batch_decay,
                        batch_norm_epsilon=RUNHEADER.m_batch_epsilon,
                        activation_fn=tf.nn.relu,
                        batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
                        batch_norm_scale=False):
    """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the inception models.
  """

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': batch_norm_updates_collections,
        # use fused batch norm if possible.
        'fused': None,
        'scale': batch_norm_scale,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay)),
                        biases_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay))):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0),
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            padding='SAME',
                            normalizer_params=normalizer_params) as sc:
            return sc
