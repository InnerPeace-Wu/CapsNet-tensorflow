# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# This file is adapted from tensorflow official tutorial of mnist.
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf
from config import cfg
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

from code.CapsNet import CapsNet

FLAGS = None


def model_test():
    model = CapsNet(None)
    model.creat_architecture()
    print("pass")


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    tf.reset_default_graph()

    # Create the model
    caps_net = CapsNet(mnist)
    caps_net.creat_architecture()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_dir = cfg.TRAIN_DIR
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if not ckpt:
        print('no checkpoint be found')

    with tf.Session(config=config) as sess:
        print("Reading parameters from %s" % ckpt.model_checkpoint_path)
        caps_net.saver.restore(sess, ckpt.model_checkpoint_path)
        for i in xrange(2000 // 30):
            caps_net.eval_reconstuct(sess, 30)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=cfg.DATA_DIR,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    # for model building test
    # model_test()
