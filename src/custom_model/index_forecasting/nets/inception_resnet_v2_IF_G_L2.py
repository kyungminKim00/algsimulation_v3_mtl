from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from custom_model.index_forecasting.common.utils import conv, linear, conv_to_fc
import header.index_forecasting.RUNHEADER as RUNHEADER
# slim = tf_slim  # call keras
import tf_slim as slim  # call tf.layers


def block_stem(inputs, scope=None, reuse=None):
    """Builds stem layer"""
    with tf.compat.v1.variable_scope(scope, 'Stem', [inputs], reuse=reuse):
        # 5 X 20 X 313
        net = slim.conv2d(inputs, 512, [1, 1], scope='Conv2d_0b_1x1')
        net = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0c_1x1')
        net = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0d_1x1')
        net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_0e_3x3')

        # 5 X 20 X 64
        with tf.compat.v1.variable_scope('Mixed_3a_layer1'):
            with tf.compat.v1.variable_scope('Branch_0_layer1'):
                branch_0 = slim.max_pool2d(net, [3, 3], padding='SAME', stride=1, scope='MaxPool_0a_3x3_layer1')
            with tf.compat.v1.variable_scope('Branch_1_layer1'):
                branch_1 = slim.conv2d(net, 96, [3, 3], scope='Conv2d_0a_3x3_layer1')
            net = tf.concat(axis=3, values=[branch_0, branch_1])
        # 5 X 20 X 160
        with tf.compat.v1.variable_scope('Mixed_4a_layer1'):
            with tf.compat.v1.variable_scope('Branch_0_layer1'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1_layer1')
                branch_0 = slim.conv2d(branch_0, 96, [3, 3], scope='Conv2d_1a_3x3_layer1')
            with tf.compat.v1.variable_scope('Branch_1_layer1'):
                branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1_layer1')
                branch_1 = slim.conv2d(branch_1, 64, [1, 5], scope='Conv2d_0b_1x7_layer1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 1], scope='Conv2d_0c_7x1_layer1')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_1a_3x3_layer1')
            net = tf.concat(axis=3, values=[branch_0, branch_1])
        # 5 x 20 x 192
        with tf.compat.v1.variable_scope('Mixed_5a_layer1'):
            with tf.compat.v1.variable_scope('Branch_0_layer1'):
                branch_0 = slim.conv2d(net, 192, [1, 3], stride=[1, 2], scope='Conv2d_1a_3x3_layer1')
            with tf.compat.v1.variable_scope('Branch_1_layer1'):
                branch_1 = slim.max_pool2d(net, [1, 3], padding='SAME', stride=[1, 2],
                                           scope='MaxPool_1a_3x3_layer1')
            net = tf.concat(axis=3, values=[branch_0, branch_1])
    return net  # 5 x 10 x 384


def block_inception(net, scope=None, reuse=None):
    """Builds stem layer"""
    with tf.compat.v1.variable_scope(scope, 'Inception', [net], reuse=reuse):
        # Inception A
        # 5 x 10 x 1152
        with tf.compat.v1.variable_scope('Mixed_4b'):
            with tf.compat.v1.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
            with tf.compat.v1.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
            with tf.compat.v1.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
            with tf.compat.v1.variable_scope('Branch_3'):
                tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
                tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
            net = tf.concat(
                [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)
        # 5 x 10 x 320 10 -> 3
        net = slim.repeat(net, 3, block5, scale=0.17, activation_fn=tf.nn.relu)

        # Inception B
        # 5 x 10 x 320
        with tf.compat.v1.variable_scope('Mixed_5a'):
            with tf.compat.v1.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 384, 3, stride=[1, 2], scope='Conv2d_1a_3x3')
            with tf.compat.v1.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, stride=[1, 2], scope='Conv2d_1a_3x3')
            with tf.compat.v1.variable_scope('Branch_2'):
                tower_pool = slim.max_pool2d(net, 3, padding='SAME', stride=[1, 2], scope='MaxPool_1a_3x3')
            net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        # 5 x 10 x 1088  9 -> 2
        net = slim.repeat(net, 2, block3, scale=0.20, activation_fn=tf.nn.relu)
        # 5 x 5 x 1088
        net = block3(net, activation_fn=None)
    return net


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
    # on - batch normalization, on - regularization
    with slim.arg_scope(inception_arg_scope(use_batch_norm=True)):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # image split
            scaled_images = tf.transpose(a=scaled_images, perm=[1, 0, 2, 3])

            scaled_images_1 = tf.transpose(a=scaled_images[0:5, :], perm=[1, 0, 2, 3])
            scaled_images_2 = tf.transpose(a=scaled_images[5:10, :], perm=[1, 0, 2, 3])
            scaled_images_3 = tf.transpose(a=scaled_images[10:15, :], perm=[1, 0, 2, 3])

            # Stem
            # 5 X 20 X 145
            layer_1_5 = block_stem(scaled_images_1, 'Stem1')
            layer_2_5 = block_stem(scaled_images_2, 'Stem2')
            layer_3_5 = block_stem(scaled_images_3, 'Stem3')

            # concat 5 x 20 x 384
            net = tf.concat([layer_1_5, layer_2_5, layer_3_5], axis=3)

            # Inception
            # 5 x 10 x 1152
            net = block_inception(net, 'Inception')

            # 5 x 5 x 1088
            net = slim.conv2d(net, 1088, 1, scope='Conv2d_5b_1x1')

            # 5 x 5 x 1088
            with tf.compat.v1.variable_scope('Feature_out'):
                kernel_size = net.get_shape()[1:3]
                if kernel_size.is_fully_defined():
                    # 5 x 5 x 1088
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a_3x3')
                # 1 x 1088
                net = slim.flatten(net)
                # 1088
                net = slim.dropout(net, RUNHEADER.m_drop_out, is_training=is_training, scope='Dropout')
                net = slim.fully_connected(net, int(512 * RUNHEADER.m_num_features),
                                           activation_fn=None, normalizer_fn=None,
                                           scope='Feature_out')

    return net  # 7 x512


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
