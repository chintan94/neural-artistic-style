## this is the main style transfer engine
import input_processor
import transfer_learning_utils as transfer_learning
import loss_functions_and_optimizer as loss_optimize
import tensorflow as tf
import numpy as np
from collections import OrderedDict

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def extract_loss(loss_store):
    return OrderedDict((key, val.eval()) for key, val in loss_store.items())


def print_loss(loss_vals):
    for key, val in loss_vals.items():
        print('{:>13s} {:g}'.format(key + ' loss:', val))


def run_style_transfer(content_image_path, style_image_path, output_image_path):
    loss_arrs = None
    content_image, style_image, iterations = input_processor.prepare_inputs_for_style_transfer(style_image_path,
                                                                                               content_image_path)

    for iteration, image, loss_values in artistic_style_transfer(
            content=content_image,
            styles=style_image,
            iterations=iterations,
    ):
        if (loss_values is not None):
            itr = []
            loss_arrs = OrderedDict((key, []) for key in loss_values.keys())
            for key, val in loss_values.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    stylized_image = input_processor.save_image(output_image_path, image)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for key, val in loss_arrs.items():
        ax.semilogy(itr, val, label=key)
    ax.legend()
    ax.set_xlabel("iterations")
    ax.set_ylabel("loss")
    plt.show()

    return stylized_image


def artistic_style_transfer(content, styles, iterations, checkpoint_iterations=50):
    """

    :param content: content image
    :param styles: style image
    :param iterations: number of iterations to run optimizer for
    :param checkpoint_iterations:
    :return: (iteration,image,loss)
    """
    shape = (1,) + content.shape
    style_shape = (1,) + styles.shape
    content_features = {}
    style_features = {}

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight

    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # defining graph for calculating content features
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, vgg_mean_pixel = transfer_learning.create_pretrained_net(image)
        content_pre = np.array([transfer_learning.pre_process_image(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # defining graph calculating style features

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net, vgg_mean_pixel = transfer_learning.create_pretrained_net(image)
        style_pre = np.array([transfer_learning.pre_process_image(styles, vgg_mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # define graph for backprop

    with tf.Graph().as_default():

        initial = tf.random_normal(shape) * 0.256
        image = tf.Variable(initial)
        net, vgg_mean_pixel = transfer_learning.create_pretrained_net(image)

        content_loss, style_loss = loss_optimize.get_content_and_style_loss(style_features, style_layers_weights,
                                                                            content_features, shape)

        loss = content_loss + style_loss

        loss_ordered_dict = OrderedDict([('content', content_loss),
                                         ('style', style_loss),
                                         ('total', loss)])

        train_step = loss_optimize.get_adam_optimizer(loss)

        best_loss = float('inf')
        best_stylized_image = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('....Starting Optimization....')

            for i in range(iterations):

                train_step.run()

                last_iteration = (i == iterations - 1)
                loss_vals = None
                if (i % checkpoint_iterations == 0) or last_iteration:
                    loss_vals = extract_loss(loss_ordered_dict)
                    print('Running iteration:' + str(i + 1))
                    print_loss(loss_vals)

                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best_stylized_image = image.eval()

                    output_image = transfer_learning.process_output(best_stylized_image.reshape(shape[1:]),
                                                                    vgg_mean_pixel)

                else:
                    output_image = None

                yield i + 1, output_image, loss_vals
