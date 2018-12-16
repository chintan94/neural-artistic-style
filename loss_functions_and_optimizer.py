## this should have all loss functions => content loss + style loss

import hyperparameters as defaults
import transfer_learning_utils as transfer_learning
import tensorflow as tf
from functools import reduce

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def get_content_and_style_loss(net, style_features, style_layers_weights, content_features, shape):
    # content loss
    content_layers_weights = {}
    content_layers_weights['relu4_2'] = 1.0
    content_layers_weights['relu5_2'] = 0.0

    content_loss = 0
    content_losses = []
    for content_layer in CONTENT_LAYERS:
        content_losses.append(content_layers_weights[content_layer] * defaults.CONTENT_WEIGHT * (2 * tf.nn.l2_loss(
            net[content_layer] - content_features[content_layer]) /
                                                                                                 content_features[
                                                                                                     content_layer].size))
    content_loss += reduce(tf.add, content_losses)

    style_loss = 0
    style_losses = []
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        _, height, width, number = map(lambda i: i.value, layer.get_shape())
        size = height * width * number
        feats = tf.reshape(layer, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(
            style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
    style_loss += defaults.STYLE_WEIGHT * reduce(tf.add, style_losses)

    return content_loss, style_loss


def get_adam_optimizer(loss):
    return tf.train.AdamOptimizer(defaults.LEARNING_RATE, defaults.BETA1, defaults.BETA2,
                                  defaults.EPSILON).minimize(loss)
