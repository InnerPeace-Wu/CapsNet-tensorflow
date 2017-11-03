# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from CapsNet import CapsNet
from config import cfg
from utils import squash, imshow_noax, tweak_matrix


def parse_arg():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train CapsNet")

    parser.add_argument('--data_dir', dest='data_dir',
                        type=str, default=cfg.DATA_DIR,
                        help='Directory for storing input data')
    parser.add_argument('--ckpt', dest='ckpt',
                        type=str, default=cfg.TRAIN_DIR,
                        help='path to the directory of check point')
    parser.add_argument('--mode', dest='mode',
                        type=str, default=None,
                        help='evaluation mode: reconstruct, cap_tweak, adversarial')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=30, help='batch size for reconstruct evaluation')
    parser.add_argument('--max_iters', dest='max_iters', type=int,
                        default=50, help='batch size for reconstruct evaluation')
    parser.add_argument('--tweak_target', dest='tweak_target', type=int,
                        default=5, help='target number for capsule tweaking experiment')
    parser.add_argument('--fig_dir', dest='fig_dir', type=str,
                        default='../figs', help='directory to save figures')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1, help='learning rate of adversarial test')

    args = parser.parse_args()

    if len(sys.argv) == 1 or \
                    args.mode not in \
                    ('reconstruct', 'cap_tweak', 'adversarial'):
        parser.print_help()
        sys.exit(1)
    return args


def main():
    args = parse_arg()

    # Import data
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    tf.reset_default_graph()

    # Create the model
    caps_net = CapsNet()
    # build up architecture
    caps_net.eval_architecture(args.mode)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # read check point file
    if args.ckpt:
        ckpt = tf.train.get_checkpoint_state(args.ckpt)
    else:
        raise ValueError, 'no ckpt found.'

    with tf.Session(config=config) as sess:
        print("Reading parameters from %s" % ckpt.model_checkpoint_path)
        caps_net.saver.restore(sess, ckpt.model_checkpoint_path)

        if args.mode == 'cap_tweak':
            save_path = args.fig_dir + '/cap_tweak'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            label = None
            while label != cfg.tweak_target:
                x, y = mnist.test.next_batch(1)
                label = np.argmax(y)
            caps_net.cap_tweak(sess, x, y, save_path)

        elif args.mode == 'reconstruct':
            save_path = args.fig_dir + '/recons'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in xrange(args.max_iters):
                x, y = mnist.test.next_batch(args.batch_size)
                caps_net.eval_reconstruct(sess, x, y, args.batch_size, save_path)





if __name__ == '__main__':
    main()
