import numpy as np
import tensorflow as tf
from custom_model.market_timing.common.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
import header.market_timing.RUNHEADER as RUNHEADER
import tf_slim as slim


def block5(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 5x5 resnet block."""
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
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up

        if activation_fn:
            net = activation_fn(net)
    return net


def block3(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 3x3 resnet block."""
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
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up

        if activation_fn:
            net = activation_fn(net)
    return net


def inception_resnet_v2(scaled_images, is_training=False, **kwargs):
    with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")):
        # image split
        scaled_images = tf.transpose(a=scaled_images, perm=[1, 0, 2, 3])
        scaled_images_1 = scaled_images[0:5, :]  # use standardization of index
        scaled_images_1 = tf.transpose(a=scaled_images_1, perm=[1, 0, 2, 3])
        scaled_images_2 = scaled_images[5:10, :]
        scaled_images_2 = tf.transpose(a=scaled_images_2, perm=[1, 0, 2, 3])
        scaled_images_3 = scaled_images[10:15, :]
        scaled_images_3 = tf.transpose(a=scaled_images_3, perm=[1, 0, 2, 3])
        # scaled_images_4 = scaled_images[15:20, :]
        # scaled_images_4 = tf.transpose(scaled_images_4, [1, 0, 2, 3])

        # Stem_1
        layer_1_1 = conv(scaled_images_1, 'Stem_1_1', n_filters=32, filter_size=3, stride=2,
                         init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        layer_1_2 = conv(layer_1_1, 'Stem_1_2', n_filters=32, filter_size=3, stride=2,
                         init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        layer_1_3 = slim.conv2d(layer_1_2, 64, 3, scope='Stem_1_3')
        layer_1_4 = slim.conv2d(layer_1_3, 96, 3, scope='Stem_1_4')


        # Stem_2
        layer_2_1 = conv(scaled_images_2, 'Stem_2_1', n_filters=32, filter_size=3, stride=2,
                         init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        layer_2_2 = conv(layer_2_1, 'Stem_2_2', n_filters=32, filter_size=3, stride=2,
                         init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        layer_2_3 = slim.conv2d(layer_2_2, 64, 3, scope='Stem_2_3')
        layer_2_4 = slim.conv2d(layer_2_3, 96, 3, scope='Stem_2_4')

        # Stem_3
        layer_3_1 = conv(scaled_images_3, 'Stem_3_1', n_filters=32, filter_size=3, stride=2,
                         init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        layer_3_2 = conv(layer_3_1, 'Stem_3_2', n_filters=32, filter_size=3, stride=2,
                         init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        layer_3_3 = slim.conv2d(layer_3_2, 64, 3, scope='Stem_3_3')
        layer_3_4 = slim.conv2d(layer_3_3, 96, 3, scope='Stem_3_4')

        # Stem_4
        # layer_4_1 = conv(scaled_images_4, 'Stem_4_1', n_filters=32, filter_size=3, stride=2,
        #                  init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        # layer_4_2 = conv(layer_4_1, 'Stem_4_2', n_filters=32, filter_size=3, stride=2,
        #                  init_scale=np.sqrt(2), pad='SAME', stride_method='Custom', **kwargs)
        # layer_4_3 = slim.conv2d(layer_4_2, 64, 3, scope='Stem_4_3')
        # layer_4_4 = slim.conv2d(layer_4_3, 96, 3, scope='Stem_4_4')

        # net = tf.concat([layer_1_4, layer_2_4, layer_3_4, layer_4_4], axis=3)
        net = tf.concat([layer_1_4, layer_2_4, layer_3_4], axis=3)

        # Inception A
        with tf.compat.v1.variable_scope('Mixed_4b'):
            with tf.compat.v1.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
            with tf.compat.v1.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                            scope='Conv2d_0b_5x5')
            with tf.compat.v1.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                            scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                            scope='Conv2d_0c_3x3')
            with tf.compat.v1.variable_scope('Branch_3'):
                tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                             scope='AvgPool_0a_3x3')
                tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                           scope='Conv2d_0b_1x1')
            net = tf.concat(
                [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

        net = slim.repeat(net, 10, block5, scale=0.17,
                          activation_fn=tf.nn.relu)

        # Inception B
        with tf.compat.v1.variable_scope('Mixed_5a'):
            with tf.compat.v1.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 384, 3, stride=2, scope='Conv2d_1a_3x3')
            with tf.compat.v1.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, stride=2, scope='Conv2d_1a_3x3')
            with tf.compat.v1.variable_scope('Branch_2'):
                tower_pool = slim.max_pool2d(net, 3, stride=1, scope='MaxPool_1a_3x3')
            net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

        net = slim.repeat(net, 9, block3, scale=0.20, activation_fn=tf.nn.relu)
        net = block3(net, activation_fn=None)

        # 3 x 3 x 804
        net = slim.conv2d(net, 804, 1, scope='Conv2d_5b_1x1')

        with tf.compat.v1.variable_scope('Feature_out'):
            kernel_size = net.get_shape()[1:3]
            if kernel_size.is_fully_defined():
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_3x3')

            net = slim.flatten(net)
            net = slim.dropout(net, 0.8, is_training=True,
                               scope='Dropout')
            net = slim.fully_connected(net, int(512*RUNHEADER.m_num_features), activation_fn=None, scope='Feature_out')



    return net


