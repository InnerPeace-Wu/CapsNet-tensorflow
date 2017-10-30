# --------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# This file is adapted from tensorflow
# official tutorial of mnist.
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from six.moves import xrange

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import slim

FLAGS = None
num_PrimaryCaps = 32
numcaps = 32*36
num_routing = 3
m_pos = 0.9
m_neg = 0.1
neg_reg = 0.5
lr = 0.001


class CapsuleMnist(object):
    def __init__(self, mnist):
        self._mnist = mnist


    def _capsule(self, input, i_c, o_c, i):
        with tf.variable_scope('cap_' + str(i)):
            w = tf.get_variable('w', shape=[9, 9, i_c, o_c], dtype=tf.float32,
                                initializer=self._def_w_initializer)
            # b = tf.get_variable('b', shape=[o_c, ], dtype=tf.float32,
            #                     initializer=self._def_b_initializer)
            cap = tf.nn.conv2d(input, w, [1, 2, 2, 1], padding='VALID', name='cap_conv')
            # cap = tf.nn.relu(cap + b)
            cap = tf.expand_dims(cap, axis=1)

            return cap

    def _squash(self, cap_s):
        s_norm = tf.norm(cap_s, ord=2, axis=2, keep_dims=True)
        s_norm_square = tf.pow(s_norm, 2)
        cap_o = tf.div(s_norm_square, 1 + s_norm_square) * \
                tf.div(cap_s, s_norm)

        return cap_o

    def _dynamic_routing(self, primary_caps):
        """"input with shape [None, 1, 32 x 6 x 6, 1, 8]"""

        with tf.name_scope('digit_caps'):
            with tf.variable_scope('digit_caps'):
                cap_ws = tf.get_variable('cap_w', shape=[10, numcaps, 8, 16], dtype=tf.float32,
                                         initializer=self._def_w_initializer)
                fn_init = tf.zeros([10, numcaps, 1, 16])
                # [None, 10, 1152, 1, 16]
                cap_predictions = tf.scan(lambda ac, x: tf.matmul(x, cap_ws),
                                     tf.tile(primary_caps, [1, 10, 1, 1, 1]), initializer=fn_init)
                # [None, 10 ,1152, 16]
                cap_predictions = tf.squeeze(cap_predictions, axis=[3])

                log_prior = tf.get_variable('log_prior', shape=[numcaps, 10], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
            for idx in xrange(num_routing):
                # [numcaps, 10]
                if idx == 0:
                    c = tf.nn.softmax(log_prior)
                    # [10, numcaps, 1]
                    c_t = tf.expand_dims(tf.transpose(c), axis=2)
                else:
                    # [None, 10, numcaps]
                    c = tf.nn.softmax(log_prior, dim=1)
                    # [None, 10, numcaps, 1]
                    c_t = tf.expand_dims(c, axis=3)

                s_t = tf.multiply(cap_predictions, c_t)
                # [None, 10, 16]
                s = tf.reduce_sum(s_t, axis=[2])
                digit_caps = self._squash(s)
                # [None, 10, 1152]
                delta_prior = tf.reduce_sum(tf.multiply(tf.expand_dims(digit_caps, axis=2),
                                                        cap_predictions),
                                            axis=[-1])
                if idx == 0:
                    log_prior = tf.transpose(log_prior)
                log_prior = log_prior + delta_prior

            return digit_caps

    def _reconstruct(self, digit_caps):
        with tf.name_scope('reconstruct'):
            y_ = tf.expand_dims(self._y_, axis=2)
            # [None, 16]
            # gt_feature = tf.reduce_sum(y_ * digit_caps, axis=1)
            # [None, 10, 16]
            gt_feature = y_ * digit_caps
            gt_feature = tf.reduce_sum(gt_feature, axis=1)
            # gt_feature = y_ * gt_feature
            fc = slim.fully_connected(gt_feature, 512,
                                      weights_initializer=self._def_w_initializer)
            fc = slim.fully_connected(fc, 1024,
                                      weights_initializer=self._def_w_initializer)
            fc = slim.fully_connected(fc, 784,
                                      weights_initializer=self._def_w_initializer,
                                      activation_fn=None)
            out = tf.sigmoid(fc)

            return out

    def _add_loss(self, digit_caps):
        with tf.name_scope('loss'):
            # [None, 10 , 1]
            # y_ = tf.expand_dims(self._y_, axis=2)
            # [None, 10]
            digit_caps_norm = tf.norm(digit_caps, ord=2, axis=2)
            # [None, ]
            loss_pos = tf.pow(tf.maximum(0., m_pos - tf.reduce_sum(digit_caps_norm * self._y_,
                                                                  axis=1),), 2)
            y_negs = 1. - self._y_
            # [None, ]
            loss_neg = neg_reg * tf.pow(tf.maximum(0., tf.reduce_sum(digit_caps_norm * y_negs,
                                                                    axis=1) - m_neg), 2)
            reconstruct = self._reconstruct(digit_caps)
            loss_resconstruct = tf.nn.l2_loss(self._x - reconstruct) * 2.
            total_loss = tf.reduce_mean(loss_pos + loss_neg + 0.0005*loss_resconstruct)

        return total_loss

    def _build_net(self):
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._y_ = tf.placeholder(tf.float32, [None, 10])
        # set up initializer for weights and bias
        self._def_w_initializer = tf.truncated_normal_initializer(stddev=0.1)
        self._def_b_initializer = tf.zeros_initializer()

        # reshape for conv ops
        with tf.name_scope('x_reshape'):
            x_image = tf.reshape(self._x, [-1, 28, 28, 1])

        # initial conv op
        with tf.variable_scope('conv1'):
            w = tf.get_variable('w', shape=[9, 9, 1, 256], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            conv1 = tf.nn.conv2d(x_image, w, [1, 1, 1, 1], padding='VALID', name='conv1')
            b = tf.get_variable('b', shape=[256, ], dtype=tf.float32,
                                initializer=self._def_b_initializer)
            conv1 = tf.nn.relu(conv1 + b)

        # build up primary capsules
        with tf.name_scope('primary_caps'):
            caps = []
            for idx in xrange(num_PrimaryCaps):
                cap = self._capsule(conv1, 256, 8, idx)
                caps.append(cap)
            # [None, 32, 6, 6, 8]
            primary_caps = tf.concat(caps, axis=1)
            # cap_shape = primary_caps.shape
            # numcaps = tf.cast(cap_shape[1] * cap_shape[2] * cap_shape[3], tf.int32)
            # [None, 32 x 6 x 6, 1, 8]
            primary_caps = tf.reshape(primary_caps, shape=[-1, 1, numcaps, 1, 8])
            self._digit_caps = self._dynamic_routing(primary_caps)
            self._loss = self._add_loss(self._digit_caps)

        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, self.global_step,
                                                   2000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self._train_op = optimizer.minimize(self._loss)
        self._accuracy()

    def _accuracy(self):
        with tf.name_scope('accuracy'):
            digit_caps_norm = tf.norm(self._digit_caps, ord=2, axis=2)
            correct_prediction = tf.equal(tf.argmax(self._y_, 1), tf.argmax(digit_caps_norm, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        self.accuracy = tf.reduce_mean(correct_prediction)

    def train_with_predict(self,sess, batch_size=50, idx = 0):
        batch = self._mnist.train.next_batch(batch_size)
        loss, _ = sess.run([self._loss, self._train_op], feed_dict={self._x: batch[0],
                                                                 self._y_: batch[1]})
        if idx % 10 == 0:
            print('accuracy: {}'.format(self.accuracy.eval(feed_dict={
                self._x: batch[0], self._y_: batch[1]})))
        return loss


def model_test():
    model = CapsuleMnist(None)
    model._build_net()
    print('pass')


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    tf.reset_default_graph()

    # Create the model
    capsule_mnist = CapsuleMnist(mnist)
    capsule_mnist._build_net()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # # Train
        for i in range(1000):
            loss = capsule_mnist.train_with_predict(sess, batch_size=50, idx=i)
            if i % 10 == 0:
                print("loss: {}".format(loss))
        # # Test trained model
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
        #                                     y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    # model_test()