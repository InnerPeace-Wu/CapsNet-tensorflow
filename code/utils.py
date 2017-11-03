# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def squash(cap_input):
    """
    squash function for keep the length of capsules between 0 - 1
    :arg
        cap_input: total input of capsules,
                   with shape: [None, h, w, c] or [None, n, d]
    :return
        cap_output: output of each capsules, which has the shape as cap_input
    """

    with tf.name_scope('squash'):
        # compute norm square of inputs with the last axis, keep dims for broadcasting
        # ||s_j||^2 in paper
        input_norm_square = tf.reduce_sum(tf.square(cap_input), axis=-1, keep_dims=True)

        # ||s_j||^2 / (1. + ||s_j||^2) * (s_j / ||s_j||)
        scale = input_norm_square / (1. + input_norm_square) / tf.sqrt(input_norm_square)

    return cap_input * scale


def imshow_noax(img, nomalize=True):
    """show image by plt with axis off"""
    if nomalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255. * (img - img_min) / (img_max - img_min)

    plt.imshow(img.astype('uint8'), cmap='gray')
    plt.gca().axis('off')


def tweak_matrix():
    """compute tweak matrix for experiment of capsule unit representations"""
    mxs = []
    t_range = np.arange(-25, 26, 5) / 100.
    id_m = np.eye(16, dtype=np.float32)
    for i in xrange(len(t_range)):
        mxs.append(id_m * t_range[i])

    tweak_m = np.concatenate(mxs, axis=0)

    return tweak_m


