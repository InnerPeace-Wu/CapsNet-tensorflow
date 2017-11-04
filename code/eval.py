# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

from CapsNet import CapsNet
from config import cfg


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
                        default=None, help='target number for capsule tweaking experiment')
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
    caps_net.eval_architecture(args.mode, args.fig_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # read check point file
    if args.ckpt:
        ckpt = tf.train.get_checkpoint_state(args.ckpt)
    else:
        raise ValueError

    with tf.Session(config=config) as sess:
        print("Reading parameters from %s" % ckpt.model_checkpoint_path)
        caps_net.saver.restore(sess, ckpt.model_checkpoint_path)

        if args.mode == 'cap_tweak':
            for i in tqdm(xrange(10), desc='capsule tweaking'):
                label = None
                while label != i:
                    x, y = mnist.test.next_batch(1)
                    label = np.argmax(y)
                caps_net.cap_tweak(sess, x, y)

        elif args.mode == 'reconstruct':
            for i in tqdm(xrange(args.max_iters), desc='reconstructing'):
                x, y = mnist.test.next_batch(args.batch_size)
                caps_net.reconstruct_eval(sess, x, y, args.batch_size)

        # adversarial test
        else:
            for ori in xrange(10):
                print('------ class {} ------'.format(ori))
                label = None
                while label != ori:
                    x, y = mnist.test.next_batch(1)
                    label = np.argmax(y)
                for tar in xrange(10):
                    if ori == tar:
                        continue
                    caps_net.adversarial_eval(sess, x, ori, tar, args.lr)


if __name__ == '__main__':
    main()
