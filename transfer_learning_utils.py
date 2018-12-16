## This contains functions to load pre-trained network

import tensorflow as tf
import numpy as np
import scipy.io as spio

## Standard layers in a VGG-19

LAYERS = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4']


## always use avg pooling
def create_pretrained_net(input_image, weight_file_path='imagenet-vgg-verydeep-19.mat'):
    '''

    loads a pre-trained VGG-19
    :return: net, mean_pixels
    '''

    vgg = spio.loadmat(weight_file_path)
    mean_channel = vgg['meta']['normalization'][0][0][0][0][2][0][0]
    weights = vgg['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][2][0]
            print(type(kernels), type(bias))
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = create_conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = create_pooling_layer(current)
        net[name] = current

    assert len(net) == len(LAYERS)
    return net, mean_channel


def pre_process_image(image, mean_channel):
    '''
    pre processing image by subtracting mean channel pixels
    :param image:
    :param mean_channel:
    :return:
    '''
    return image - mean_channel


def process_output(image, mean_channel):
    '''
    add back mean pixels to output images
    :param image:
    :param mean_channel:
    :return:
    '''

    return image + mean_channel


def create_conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def create_pooling_layer(input):
    return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')
