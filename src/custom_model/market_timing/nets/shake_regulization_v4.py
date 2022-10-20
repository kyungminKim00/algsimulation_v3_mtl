from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from custom_model.market_timing.common.utils import conv, linear, conv_to_fc
import header.market_timing.RUNHEADER as RUNHEADER
# slim = tf_slim  # call keras
import tf_slim as slim  # call tf.layers


def ShakeShake(x, is_training=False):
    """ Shake-Shake-Image Layer """
    # unpack x1 and x2
    assert isinstance(x, list)
    x1, x2 = x
    # create alpha and beta
    batch_size = tf.shape(input=x1)[0]
    alpha = tf.random.uniform((batch_size, 1, 1, 1))
    beta = tf.random.uniform((batch_size, 1, 1, 1))

    # shake-shake during training phase
    def x_shake():
        return beta * x1 + (1 - beta) * x2 + tf.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)

    # even-even during testing phase
    def x_even():
        return 0.5 * x1 + 0.5 * x2

    # return K.in_train_phase(x_shake, x_even)
    shake_layer = tf.case([(tf.equal(is_training, True), x_shake)], default=x_even)
    return shake_layer


def block_stem(inputs, scope=None, reuse=None):
    """Builds stem layer"""
    with tf.compat.v1.variable_scope(scope, 'Stem', [inputs], reuse=reuse):
        # 5 X 20 X 150
        net = cbam_block(inputs, 'CBAM_a', ratio=8)
        # net = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0b_1x1')
        # net = slim.conv2d(net, 21, [3, 3], scope='Conv2d_0c_1x1')
        net = slim.conv2d(net, 16, [1, 1], activation_fn=tf.nn.relu, scope='Conv2d_0b_1x1')
        net = slim.conv2d(net, 21, [3, 3], activation_fn=tf.nn.relu, scope='Conv2d_0c_1x1')
        net = cbam_block(net, 'CBAM_b', ratio=8)
        net = slim.max_pool2d(net, kernel_size=3, stride=1, padding='SAME')

    return net  # 5 x 20 x 21


def create_residual_branch(net, filters, stride):
    """ Regular Branch of a Residual network: ReLU -> Conv2D -> BN repeated twice """
    net = tf.nn.relu(net)
    net = slim.conv2d(net, filters, kernel_size=3, stride=stride)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, filters, kernel_size=3, stride=1)
    return net


def create_residual_shortcut(net, filters, stride, is_training=False):
    """ Shortcut Branch used when downsampling from Shake-Shake regularization """
    net = tf.nn.relu(net)
    h_stride, w_stride = stride

    x1 = net[:, 0::h_stride, 0::w_stride, :]
    x2 = net[:, 0::h_stride, 1::w_stride, :]
    x1 = slim.conv2d(x1, filters // 2, kernel_size=1, stride=1, padding='valid',
                     normalizer_fn=None, normalizer_params={})
    x2 = slim.conv2d(x2, filters // 2, kernel_size=1, stride=1, padding='valid',
                     normalizer_fn=None, normalizer_params={})

    net = tf.concat([x1, x2], axis=3)
    net = slim.batch_norm(net, decay=RUNHEADER.m_batch_decay, epsilon=RUNHEADER.m_batch_epsilon,
                          is_training=is_training)
    return net


def create_residual_shortcut2(net, filters, stride, is_training=False):
    """ Shortcut Branch used when downsampling from Shake-Shake regularization """
    net = tf.nn.relu(net)
    h_stride, w_stride = stride

    x1 = net[:, 0::h_stride, 0::w_stride, :]
    x2 = net[:, 1:-1:1, 1::w_stride, :]
    x1 = slim.conv2d(x1, filters // 2, kernel_size=1, stride=1, padding='valid',
                     normalizer_fn=None, normalizer_params={})
    x2 = slim.conv2d(x2, filters // 2, kernel_size=1, stride=1, padding='valid',
                     normalizer_fn=None, normalizer_params={})

    net = tf.concat([x1, x2], axis=3)
    net = slim.batch_norm(net, decay=RUNHEADER.m_batch_decay, epsilon=RUNHEADER.m_batch_epsilon,
                          is_training=is_training)
    return net


def create_residual_block(net, filters, stride=None, is_training=False):
    """ Residual Block with Shake-Shake regularization and shortcut """
    x1 = create_residual_branch(net, filters, stride)
    x2 = create_residual_branch(net, filters, stride)
    if stride[0] == 1 and stride[1] == 2:
        net = create_residual_shortcut(net, filters, stride, is_training=is_training)
    elif stride[0] == 2 and stride[1] == 2:
        net = create_residual_shortcut2(net, filters, stride, is_training=is_training)
    elif stride[0] == 2 and stride[1] == 1:
        assert True, 'Not defined yet'
    elif stride[0] == 1 and stride[1] == 1:
        pass
    else:
        assert True, 'Not defined yet'
    # return keras.layers.Add()([net, ShakeShake()([x1, x2])])
    block = ShakeShake([x1, x2], is_training=is_training)
    net += block
    return net


def block_residual_layer(net, filters, blocks, stride, is_training=False):
    net = create_residual_block(net, filters, stride, is_training=is_training)
    for i in range(1, blocks):
        net = create_residual_block(net, filters, [1, 1], is_training=is_training)
    return net


def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.compat.v1.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
    return attention_feature


def channel_attention(input_feature, name, ratio=8):

    with tf.compat.v1.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_tensor=input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.compat.v1.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0),
                                   bias_initializer=tf.compat.v1.constant_initializer(value=0.0),
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.compat.v1.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0),
                                   bias_initializer=tf.compat.v1.constant_initializer(value=0.0),
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_tensor=input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.compat.v1.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.compat.v1.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    # kernel_size = 7
    kernel_size = 3
    with tf.compat.v1.variable_scope(name):
        avg_pool = tf.reduce_mean(input_tensor=input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_tensor=input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.compat.v1.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0),
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat


def shakenet(scaled_images, is_training=False, **kwargs):
    # on - batch normalization, on - regularization
    with slim.arg_scope(shake_arg_scope(use_batch_norm=True)):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # image split
            scaled_images = tf.transpose(a=scaled_images, perm=[1, 0, 2, 3])

            scaled_images_1 = tf.transpose(a=scaled_images[0:5, :], perm=[1, 0, 2, 3])
            scaled_images_2 = tf.transpose(a=scaled_images[5:10, :], perm=[1, 0, 2, 3])
            scaled_images_3 = tf.transpose(a=scaled_images[10:15, :], perm=[1, 0, 2, 3])

            # Stem
            # 5 X 20 X 150
            layer_1_5 = block_stem(scaled_images_1, 'Stem1')
            layer_2_5 = block_stem(scaled_images_2, 'Stem2')
            layer_3_5 = block_stem(scaled_images_3, 'Stem3')

            # concat 5 x 20 x 63
            net = tf.concat([layer_1_5, layer_2_5, layer_3_5], axis=3)
            net = cbam_block(net, 'CBAM_a', ratio=8)

            # 5 X 20 X 63
            net = block_residual_layer(net, 63, 5, [1, 1], is_training=is_training)
            # 5 X 20 X 63
            net = block_residual_layer(net, 128, 5, [1, 2], is_training=is_training)
            # 5 X 10 X 128
            net = block_residual_layer(net, 256, 5, [2, 2], is_training=is_training)

            # 3 X 5 X 256
            net = tf.nn.relu(net)

            # 5 x 5 x 1088
            with tf.compat.v1.variable_scope('Feature_out'):
                kernel_size = net.get_shape()[1:3]
                if kernel_size.is_fully_defined():
                    # 3 X 5 X 256
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a_3x3')
                # 1 X 1 x 256
                net = slim.flatten(net)
                # 256
                net = slim.dropout(net, RUNHEADER.m_drop_out, is_training=is_training, scope='Dropout')
                net = slim.fully_connected(net, int(512 * RUNHEADER.m_num_features),
                                           activation_fn=None, normalizer_fn=None,
                                           scope='Feature_out')

    return net  # 512


def shake_arg_scope(weight_decay=RUNHEADER.m_l2_norm,
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
                        biases_regularizer=None):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0),
                            biases_initializer=None,
                            activation_fn=None,
                            normalizer_fn=normalizer_fn,
                            padding='SAME',
                            normalizer_params=normalizer_params) as sc:
            return sc
