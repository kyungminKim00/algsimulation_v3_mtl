import numpy as np
import tensorflow as tf
from custom_model.market_timing.common.utils import conv, linear, conv_to_fc
import header.market_timing.RUNHEADER as RUNHEADER


def nature_cnn(scaled_images, is_training=False, **kwargs):
    activ = tf.nn.relu
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    # image split
    scaled_images = tf.transpose(a=scaled_images, perm=[1, 0, 2, 3])

    scaled_images_1 = scaled_images[0:5, :]  # use standardization of index
    scaled_images_1 = tf.transpose(a=scaled_images_1, perm=[1, 0, 2, 3])
    scaled_images_2 = scaled_images[5:10, :]
    scaled_images_2 = tf.transpose(a=scaled_images_2, perm=[1, 0, 2, 3])
    scaled_images_3 = scaled_images[10:15, :]
    scaled_images_3 = tf.transpose(a=scaled_images_3, perm=[1, 0, 2, 3])
    
    

    layer_1_1 = conv(scaled_images_1, 'c1_1', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                     **kwargs)
    layer_1_1 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_1_1, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_1_1 = activ(layer_1_1)
    layer_2_1 = activ(conv(layer_1_1, 'c2_1', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                           stride_method='Custom', **kwargs))
    layer_2_1 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_2_1, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_2_1 = activ(layer_2_1)
    layer_3_1 = activ(conv(layer_2_1, 'c3_1', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                           stride_method='Custom', **kwargs))
    layer_3_1 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_3_1, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_3_1 = activ(layer_3_1)

    layer_1_2 = conv(scaled_images_2, 'c1_2', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                     **kwargs)
    layer_1_2 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_1_2, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_1_2 = activ(layer_1_2)
    layer_2_2 = activ(conv(layer_1_2, 'c2_2', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                           stride_method='Custom', **kwargs))
    layer_2_2 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_2_2, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_2_2 = activ(layer_2_2)
    layer_3_2 = activ(conv(layer_2_2, 'c3_2', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                           stride_method='Custom', **kwargs))
    layer_3_2 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_3_2, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_3_2 = activ(layer_3_2)

    layer_1_3 = conv(scaled_images_3, 'c1_3', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                     **kwargs)
    layer_1_3 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_1_3, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_1_3 = activ(layer_1_3)
    layer_2_3 = activ(conv(layer_1_3, 'c2_3', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                           stride_method='Custom', **kwargs))
    layer_2_3 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_2_3, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_2_3 = activ(layer_2_3)
    layer_3_3 = activ(conv(layer_2_3, 'c3_3', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME', use_BN=True,
                           stride_method='Custom', **kwargs))
    layer_3_3 = tf.compat.v1.layers.batch_normalization(
        inputs=layer_3_3, momentum=0.997, epsilon=1e-5, training=is_training, scale=False)
    layer_3_3 = activ(layer_3_3)

    # layer_1_4 = conv(scaled_images_4, 'c1_4', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME',
    #                  **kwargs)
    # layer_1_4 = tf.layers.batch_normalization(
    #     inputs=layer_1_4, momentum=0.997, epsilon=1e-5, training=is_training)
    # layer_1_4 = activ(layer_1_4)
    # layer_2_4 = activ(conv(layer_1_4, 'c2_4', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME',
    #                        stride_method='Custom', **kwargs))
    # layer_2_4 = tf.layers.batch_normalization(
    #     inputs=layer_2_4, momentum=0.997, epsilon=1e-5, training=is_training)
    # layer_2_4 = activ(layer_2_4)
    # layer_3_4 = activ(conv(layer_2_4, 'c3_4', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), pad='SAME',
    #                        stride_method='Custom', **kwargs))
    # layer_3_4 = tf.layers.batch_normalization(
    #     inputs=layer_3_4, momentum=0.997, epsilon=1e-5, training=is_training)
    # layer_3_4 = activ(layer_3_4)

    # layer_4 = tf.concat([layer_3_1, layer_3_2, layer_3_3, layer_3_4], axis=3)
    layer_4 = tf.concat([layer_3_1, layer_3_2, layer_3_3], axis=3)
    layer_4 = tf.compat.v1.layers.average_pooling2d(layer_4,
                                          [layer_4.get_shape().as_list()[1], layer_4.get_shape().as_list()[2]],
                                          [layer_4.get_shape().as_list()[1], layer_4.get_shape().as_list()[2]],
                                          'SAME')
    layer_4 = conv_to_fc(layer_4)

    output = linear(layer_4, 'fc1', n_hidden=int(512 * RUNHEADER.m_num_features), init_scale=np.sqrt(2))
    output = activ(output)

    # original
    # output = linear(layer_4, 'fc1', n_hidden=int(512 * RUNHEADER.m_num_features), init_scale=np.sqrt(2))
    # output = activ(output)

    return output
