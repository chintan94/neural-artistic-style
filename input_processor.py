## This file reads content and style images and also creates hyperparameters for Adam optimizer#

import os
import math
import re
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import numpy as np
import scipy.misc

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'avg'



def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def get_default_hyperparameters_for_adam():
    '''
    return beta1,beta2,....
    :return:
    '''


def prepare_inputs_for_style_transfer(style_image_path, content_image_path, iterations=1000):
    '''

    returns all data needed for style_transfer(stylize)
    :return:
    '''

    content_image = imread(content_image_path)
    style_image = imread(style_image_path)

    return content_image, style_image, iterations
