## This file reads content and style images and also creates hyperparameters for Adam optimizer#

from PIL import Image
import numpy as np
import scipy.misc


def read_image(path):
    img = scipy.misc.imread(path).astype(np.float32)
    return img


def save_image(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)



def prepare_inputs_for_style_transfer(style_image_path, content_image_path):
    '''
    reads content and style images for style transfer
    returns all data needed for style_transfer
    :return:
    '''

    content_image = read_image(content_image_path)
    style_image = read_image(style_image_path)

    return content_image, style_image
