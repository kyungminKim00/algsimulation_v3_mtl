import numpy as np
import tensorflow as tf
from custom_model.fund_selection.common.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
import header.fund_selection.RUNHEADER as RUNHEADER
import tf_slim as slim


def nature_cnn(scaled_images, is_training = False, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    # Add BatchNormal - 2019-08-06
    activ = tf.nn.relu

    layer_1 = conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs)
    layer_1 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_1, momentum=0.997, epsilon=1e-5, fused=True, training=is_training)
    layer_1 = activ(layer_1)

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_2, momentum=0.997, epsilon=1e-5, fused=True, training=is_training)
    layer_2 = activ(layer_2)

    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_3, momentum=0.997, epsilon=1e-5, fused=True, training=is_training)
    layer_3 = activ(layer_3)

    layer_3 = conv_to_fc(layer_3)
    output = linear(layer_3, 'fc1', n_hidden=int(512 * RUNHEADER.m_num_features), init_scale=np.sqrt(2))
    output = activ(output)

    return output
