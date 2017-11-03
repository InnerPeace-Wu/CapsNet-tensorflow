# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow.contrib import slim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import squash, imshow_noax, tweak_matrix
from config import cfg


class CapsNet(object):
    def __init__(self):
        """initial class with mnist dataset"""
        # keep tracking of the dimension of feature maps
        self._dim = 28
        # store number of capsules of each capsule layer
        # the conv1-layer has 0 capsules
        self._num_caps = [0]
        # set for counting
        self._count = 0
        # set up placeholder of input data and labels
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._y_ = tf.placeholder(tf.float32, [None, 10])
        # set up initializer for weights and bias
        self._w_initializer = tf.truncated_normal_initializer(stddev=0.1)
        self._b_initializer = tf.zeros_initializer()

    def _capsule(self, input, i_c, o_c, idx):
        """
        compute a capsule,
        conv op with kernel: 9x9, stride: 2,
        padding: VALID, output channels: 8 per capsule.
        As described in the paper.
        :arg
            input: input for computing capsule, shape: [None, w, h, c]
            i_c: input channels
            o_c: output channels
            idx: index of the capsule about to create

        :return
            capsule: computed capsule
        """
        with tf.variable_scope('cap_' + str(idx)):
            w = tf.get_variable('w', shape=[9, 9, i_c, o_c], dtype=tf.float32)
            cap = tf.nn.conv2d(input, w, [1, 2, 2, 1],
                               padding='VALID', name='cap_conv')
            if cfg.USE_BIAS:
                b = tf.get_variable('b', shape=[o_c, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                cap = cap + b
            # cap with shape [None, 6, 6, 8] for mnist dataset

            # Note: use "squash" as its non-linearity.
            capsule = squash(cap)
            # capsule with shape: [None, 6, 6, 8]
            # expand the dimensions to [None, 1, 6, 6, 8] for following concat
            capsule = tf.expand_dims(capsule, axis=1)

            # return capsule with shape [None, 1, 6, 6, 8]
            return capsule

    def _dynamic_routing(self, primary_caps, layer_index):
        """"
        dynamic routing between capsules
        :arg
            primary_caps: primary capsules with shape [None, 1, 32 x 6 x 6, 1, 8]
            layer_index: index of the current capsule layer, i.e. the input layer for routing
        :return
            digit_caps: the output of digit capsule layer output, with shape: [None, 10, 16]
        """
        # number of the capsules in current layer
        num_caps = self._num_caps[layer_index]
        # weight matrix for capsules in "layer_index" layer
        # W_ij
        cap_ws = tf.get_variable('cap_w', shape=[10, num_caps, 8, 16],
                                 dtype=tf.float32,
                                 )
        # initial value for "tf.scan", see official doc for details
        fn_init = tf.zeros([10, num_caps, 1, 16])

        # x after tiled with shape: [10, num_caps, 1, 8]
        # cap_ws with shape: [10, num_caps, 8, 16],
        # [8 x 16] for each pair of capsules between two layers
        # u_hat_j|i = W_ij * u_i
        cap_predicts = tf.scan(lambda ac, x: tf.matmul(x, cap_ws),
                               tf.tile(primary_caps, [1, 10, 1, 1, 1]),
                               initializer=fn_init, name='cap_predicts')
        # cap_predicts with shape: [None, 10, num_caps, 1, 16]
        cap_predictions = tf.squeeze(cap_predicts, axis=[3])
        # after squeeze with shape: [None, 10, num_caps, 16]

        # log prior probabilities
        log_prior = tf.get_variable('log_prior', shape=[10, num_caps], dtype=tf.float32,
                                    initializer=tf.zeros_initializer(),
                                    trainable=cfg.PRIOR_TRAINING)
        # log_prior with shape: [10, num_caps]
        # V1. static way
        if cfg.ROUTING_WAY == 'static':
            digit_caps = self._dynamic_routingV1(log_prior, cap_predictions)
        # V2. dynamic way
        elif cfg.ROUTING_WAY == 'dynamic':
            digit_caps = self._dynamic_routingV2(log_prior, cap_predictions, num_caps)
        else:
            raise NotImplementedError

        return digit_caps

    def _dynamic_routingV2(self, prior, cap_predictions, num_caps):
        """
        doing dynamic routing with tf.while_loop
        :arg
            proir: log prior for scaling with shape [10, num_caps]
            cap_prediction: predictions from layer below with shape [None, 10, num_caps, 16]
            num_caps: num_caps
        :return
            digit_caps: digit capsules with shape [None, 10, 16]
        """
        # see issue for more info: https://github.com/XifengGuo/CapsNet-Keras/issues/1
        # check V1 for implementation details
        init_cap = tf.reduce_sum(cap_predictions, -2)
        iters = tf.constant(cfg.ROUTING_ITERS)
        prior = tf.expand_dims(prior, 0)

        def body(i, prior, cap_out):
            c = tf.nn.softmax(prior, dim=1)
            c_expand = tf.expand_dims(c, axis=-1)
            s_t = tf.multiply(cap_predictions, c_expand)
            s = tf.reduce_sum(s_t, axis=[2])
            cap_out = squash(s)
            delta_prior = tf.reduce_sum(tf.multiply(tf.expand_dims(cap_out, axis=2),
                                                    cap_predictions),
                                        axis=[-1])
            prior = prior + delta_prior

            return [i - 1, prior, cap_out]

        condition = lambda i, proir, cap_out: i > 0
        _, prior, digit_caps = tf.while_loop(condition, body, [iters, prior, init_cap],
                                             shape_invariants=[iters.get_shape(),
                                                               tf.TensorShape([None, 10, num_caps]),
                                                               init_cap.get_shape()])

        return digit_caps

    def _dynamic_routingV1(self, prior, cap_predictions):
        """
        doing dynamic routing with for loop as static implementation
        :arg
            proir: log prior for scaling with shape [10, num_caps]
            cap_prediction: predictions from layer below with shape [None, 10, num_caps, 16]
        :return
            digit_caps: digit capsules with shape [None, 10, 16]
        """
        prior = tf.expand_dims(prior, 0)
        # prior shape: [1, 10, num_caps]
        for idx in xrange(cfg.ROUTING_ITERS):
            with tf.name_scope('routing_%s' % idx):
                c = tf.nn.softmax(prior, dim=1)
                # c shape: [1, 10, num_caps]
                c_t = tf.expand_dims(c, axis=-1)
                # c_t shape: [1, 10, num_caps, 1]

                s_t = tf.multiply(cap_predictions, c_t)
                # s_t shape: [None, 10, num_caps, 16]
                # for each capsule in the layer after, add all the weighted capsules to get
                # the capsule input for it.
                # s_j = Sum_i (c_ij u_hat_j|i)
                s = tf.reduce_sum(s_t, axis=[2])

                # s shape: [None, 10, 16]
                digit_caps = squash(s)
                # digit_caps shape: [None, 10, 16]

                # u_hat_j|i * v_j
                delta_prior = tf.reduce_sum(tf.multiply(tf.expand_dims(digit_caps, axis=2),
                                                        cap_predictions),
                                            axis=[-1])
                # delta_prior shape: [None, 10, num_caps]

                prior = prior + delta_prior

        # shape [None, 10, 16]
        return digit_caps

    def _reconstruct(self, target_cap):
        """
        reconstruct from digit capsules with 3 fully connected layer
        :param
            digit_caps: digit capsules with shape [None, 16]
        :return:
            out: out of reconstruction
        """

        with tf.name_scope('reconstruct'):
            fc = slim.fully_connected(target_cap, 512,
                                      weights_initializer=self._w_initializer)
            fc = slim.fully_connected(fc, 1024,
                                      weights_initializer=self._w_initializer)
            fc = slim.fully_connected(fc, 784,
                                      weights_initializer=self._w_initializer,
                                      activation_fn=None)
            # the last layer with sigmoid activation
            out = tf.sigmoid(fc)
            # out with shape [None, 784]

            self._recons_img = out
            return out

    def _add_loss(self):
        """
        add the margin loss and reconstruction loss
        :arg
            digit_caps: output of digit capsule layer, shape [None, 10, 16]
        :return
            total_loss:
        """
        with tf.name_scope('loss'):
            # [None, 10 , 1]
            # y_ = tf.expand_dims(self._y_, axis=2)

            # loss of positive classes
            # max(0, m+ - ||v_c||) ^ 2
            with tf.name_scope('pos_loss'):
                pos_loss = tf.maximum(0., cfg.M_POS - tf.reduce_sum(self._digit_caps_norm * self._y_,
                                                                    axis=1), name='pos_max')
                pos_loss = tf.square(pos_loss, name='pos_square')
                pos_loss = tf.reduce_mean(pos_loss)
            tf.summary.scalar('pos_loss', pos_loss)
            # pos_loss shape: [None, ]

            # get index of negative classes
            y_negs = 1. - self._y_
            # max(0, ||v_c|| - m-) ^ 2
            with tf.name_scope('neg_loss'):
                neg_loss = tf.maximum(0., self._digit_caps_norm * y_negs - cfg.M_NEG)
                neg_loss = tf.reduce_sum(tf.square(neg_loss), axis=-1) * cfg.LAMBDA
                neg_loss = tf.reduce_mean(neg_loss)
            tf.summary.scalar('neg_loss', neg_loss)
            # neg_loss shape: [None, ]

            # (TODO wu) there is two ways to do reconstruction.
            # 1. only use the target capsule with dimension [None, 16] or [16,] (use it for default)
            # 2. use all the capsule, including the masked out ones with lots of zeros

            y_ = tf.expand_dims(self._y_, axis=2)
            # y_ shape: [None, 10, 1]

            # for method 1.
            target_cap = y_ * self._digit_caps
            # target_cap shape: [None, 10, 16]
            target_cap = tf.reduce_sum(target_cap, axis=1)
            # target_cap: [None, 16]

            # for method 2.
            # target_cap = tf.reshape(y_ * digit_caps, [-1, 10*16])

            reconstruct = self._reconstruct(target_cap)

            # loss of reconstruction
            with tf.name_scope('l2_loss'):
                reconstruct_loss = tf.reduce_sum(tf.square(self._x - reconstruct), axis=-1)
                reconstruct_loss = tf.reduce_mean(reconstruct_loss)
            tf.summary.scalar('reconstruct_loss', reconstruct_loss)

            total_loss = pos_loss + neg_loss + \
                         cfg.RECONSTRUCT_W * reconstruct_loss

            tf.summary.scalar('loss', total_loss)

        return total_loss

    def train_architecture(self):
        """creat architecture of the whole network"""

        with tf.variable_scope('CapsNet', initializer=self._w_initializer):
            # build net
            self._build_net()

            # set up losses
            self._loss = self._add_loss()

            # set up exponentially decay learning rate
            self._global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(cfg.LR, self._global_step,
                                                       cfg.STEP_SIZE, cfg.DECAY_RATIO,
                                                       staircase=True)
            tf.summary.scalar('learning rate', learning_rate)

            # set up adam optimizer with default setting
            self._optimizer = tf.train.AdamOptimizer(learning_rate)
            gradidents = self._optimizer.compute_gradients(self._loss)
            tf.summary.scalar('grad_norm', tf.global_norm(gradidents))

            self._train_op = self._optimizer.apply_gradients(gradidents,
                                                             global_step=self._global_step)
            # set up accuracy ops
            self._accuracy()
            # set up summary op
            self._summary_op = tf.summary.merge_all()
            # create a saver
            self.saver = tf.train.Saver()

            # set up summary writer
            self.train_writer = tf.summary.FileWriter(cfg.TB_DIR + '/train')
            self.val_writer = tf.summary.FileWriter(cfg.TB_DIR + '/val')

    def _build_net(self):
        """build the graph of the network"""

        # reshape for conv ops
        with tf.name_scope('x_reshape'):
            x_image = tf.reshape(self._x, [-1, 28, 28, 1])

        # initial conv1 op
        # 1). conv1 with kernel 9x9, stride 1, output channels 256
        with tf.variable_scope('conv1'):
            # specially initialize it with xavier initializer with no good reason.
            w = tf.get_variable('w', shape=[9, 9, 1, 256], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            # conv op
            conv1 = tf.nn.conv2d(x_image, w, [1, 1, 1, 1],
                                 padding='VALID', name='conv1')
            if cfg.USE_BIAS:
                # bias (TODO wu) no idea if the paper uses bias or not
                b = tf.get_variable('b', shape=[256, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                conv1 = tf.nn.relu(conv1 + b)
            else:
                conv1 = tf.nn.relu(conv1)

            # update dimensions of feature map
            self._dim = (self._dim - 9) // 1 + 1
            assert self._dim == 20, "after conv1, dimensions of feature map" \
                                    "should be 20x20"

            # conv1 with shape [None, 20, 20, 256]

        # build up primary capsules
        with tf.variable_scope('PrimaryCaps'):

            # update dim of capsule grid
            self._dim = (self._dim - 9) // 2 + 1
            # number of primary caps: 6x6x32 = 1152
            self._num_caps.append(self._dim ** 2 * cfg.PRIMARY_CAPS_CHANNELS)
            assert self._dim == 6, "dims for primary caps grid should be 6x6."

            # build up PriamryCaps with 32 channels and 8-D vector
            # 1. dummy solution
            '''
            caps = []
            for idx in xrange(cfg.PRIMARY_CAPS_CHANNELS):
                # get a capsule with 8-D
                cap = self._capsule(conv1, 256, 8, idx)
                # cap with shape: [None, 1, 6, 6, 8]
                caps.append(cap)

            # concat all the primary capsules
            primary_caps = tf.concat(caps, axis=1)
            # primary_caps with shape: [None, 32, 6, 6, 8]
            with tf.name_scope('primary_cap_reshape'):
                # reshape and expand dims for broadcasting in dynamic routing
                primary_caps = tf.reshape(primary_caps,
                                                  shape=[-1, 1, self._num_caps[1], 1, 8])
                # primary_caps with shape: [None, 1, 1152, 1, 8]
            '''
            # 2. faster one
            primary_caps = slim.conv2d(conv1, 32 * 8, 9, 2, padding='VALID', activation_fn=None)
            primary_caps = tf.reshape(primary_caps, [-1, 1, self._num_caps[1], 1, 8])
            primary_caps = squash(primary_caps)

        # dynamic routing
        with tf.variable_scope("digit_caps"):
            self._digit_caps = self._dynamic_routing(primary_caps, 1)

            self._digit_caps_norm = tf.norm(self._digit_caps, ord=2, axis=2,
                                            name='digit_caps_norm')
            # digit_caps_norm shape: [None, 10]

    def _accuracy(self):
        """set up accuracy"""
        with tf.name_scope('accuracy'):
            # digit_caps_norm = tf.norm(self._digit_caps, ord=2, axis=-1)
            self._py = tf.argmax(self._digit_caps_norm, 1)
            correct_prediction = tf.equal(tf.argmax(self._y_, 1),
                                          self._py)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy', self.accuracy)

    def get_summary(self, sess, data):
        """get training summary"""

        acc, summary = sess.run([self.accuracy, self._summary_op],
                                feed_dict={self._x: data[0],
                                           self._y_: data[1]})

        return acc, summary

    def train(self, sess, data):
        """training process
        :arg
            data: (images, labels)
        """
        loss, _ = sess.run([self._loss, self._train_op],
                           feed_dict={self._x: data[0],
                                      self._y_: data[1]})

        return loss

    def snapshot(self, sess, iters):
        """save checkpoint"""

        save_path = cfg.TRAIN_DIR + '/capsnet'
        self.saver.save(sess, save_path, iters)

    def test(self, sess, mnist, set='validation'):
        """test trained model on specific dataset split"""
        tic = time.time()
        if set == 'test':
            x = mnist.test.images
            y_ = mnist.test.labels
        elif set == 'validation':
            x = mnist.validation.images
            y_ = mnist.validation.labels
        elif set == 'train':
            x = mnist.train.images
            y_ = mnist.train.labels
        else:
            raise ValueError

        acc = []
        for i in tqdm(xrange(len(x) // 100), desc="calculating %s accuracy" % set):
            x_i = x[i * 100: (i + 1) * 100]
            y_i = y_[i * 100: (i + 1) * 100]
            ac = sess.run(self.accuracy,
                          feed_dict={self._x: x_i, self._y_: y_i})
            acc.append(ac)

        all_ac = np.mean(np.array(acc))
        t = time.time() - tic
        print("{} set accuracy: {}, with time: {:.2f} secs".format(set, all_ac, t))

    def eval_reconstuct(self, sess, x, y, batch_size, save_path):
        """do image reconstruction and representations"""
        ori_img = x
        label = np.argmax(y, axis=1)
        res_img, res_label, acc, norms = sess.run([self._recons_img, self._py, self.accuracy,
                                                   self._digit_caps_norm],
                                                  feed_dict={self._x: x,
                                                             self._y_: y})
        if acc < 1:
            ori_img = np.reshape(ori_img, [batch_size, 28, 28])
            res_img = np.reshape(res_img, [batch_size, 28, 28])
            num_rows = int(np.ceil(batch_size / 10))
            fig, _ = plt.subplots(nrows=2 * num_rows, ncols=10, figsize=(10, 7))
            for r in xrange(num_rows):
                for i in xrange(10):
                    idx = i + r * 10
                    if idx == batch_size:
                        break
                    plt.subplot(2 * num_rows, 10, idx + 1 + 10 * r)
                    imshow_noax(ori_img[idx])
                    plt.title(label[idx])
                    plt.subplot(2 * num_rows, 10, idx + 11 + 10 * r)
                    imshow_noax(res_img[idx])
                    plt.title("%s_%.3f" % (res_label[idx], norms[idx][res_label[idx]]))

            # plt.show()
            self._count += 1
            plt.savefig(save_path + '/%s.png' % self._count, dpi=200)

    def eval_architecture(self, mode):
        """evaluation architecture"""
        with tf.variable_scope('CapsNet', initializer=self._w_initializer):
            self._build_net()
            if mode in ('cap_tweak', 'reconstruct'):
                y_expand = tf.expand_dims(self._y_, axis=2)
                target_cap = tf.multiply(self._digit_caps, y_expand)
                # [None, 16]
                target_cap = tf.reduce_sum(target_cap, axis=1)
                if mode == 'cap_tweak':
                    # [None, 1, 16]
                    target_cap = tf.expand_dims(target_cap, axis=1)
                    # [None, 176, 16]
                    caps = target_cap + tweak_matrix()
                    self._reconstruct(tf.reshape(caps, [-1, 16]))
                else:
                    self._reconstruct(target_cap)
            elif mode == 'adversarial':
                self._adver_loss = tf.reduce_sum(self._digit_caps_norm * self._y_)
                grads = tf.gradients(self._adver_loss, self._x)
                self._adver_graidents = grads / tf.norm(grads)
                self._py = tf.argmax(self._digit_caps_norm, 1)
            else:
                raise NotImplementedError

            self.saver = tf.train.Saver()

    def cap_tweak(self, sess, x, y, save_path='../figs/cap_tweak'):

        res_img = sess.run(self._recons_img, feed_dict={self._x: x,
                                                        self._y_: y})
        res_img = np.reshape(res_img, [-1, 28, 28])
        fig, _ = plt.subplots(nrows=11, ncols=16, figsize=(10, 7))
        for i in xrange(11):
            for j in xrange(16):
                idx = j + i * 16
                plt.subplot(11, 16, idx + 1)
                imshow_noax(res_img[idx])

        # plt.show()
        plt.savefig(save_path + '/dr_exp_%s.png' % self._count, dpi=200)
        self._count += 1

    def adversarial_test(self, sess, ori_num, target_num, lamb=1):
        """advertisal test"""
        label = [None]
        while label[0] != ori_num:
            data = self._mnist.test.next_batch(1)
            label = np.argmax(np.array(data[1]), axis=1)
        count = 0
        ori_img = np.reshape(data[0], [-1, 28, 28])
        x = data[0].copy()
        py = None

        tar_oh = np.zeros(10, dtype=np.float32)
        tar_oh[target_num] = 1
        while py != target_num:
            grads, py, norm = sess.run([self._adver_graidents, self._py, self._adver_graidents_norm],
                                       feed_dict={self._x: x,
                                                  self._y_: tar_oh[None, :]})
            x += lamb * grads[0]
            count += 1
            print("predict: {}, count: {}".format(py, count))
            print("diff: {}".format(np.sum((x - data[0]))))

        x = np.reshape(x, [-1, 28, 28])

        plt.subplot(1, 3, 1)
        imshow_noax(ori_img[0])
        plt.title('orignal: %s' % label[0])
        plt.subplot(1, 3, 2)
        imshow_noax(x[0])
        plt.title('adversarial: %s' % target_num)
        plt.subplot(1, 3, 3)
        imshow_noax((x[0] - ori_img[0]))
        plt.title('difference(normalized)')

        plt.savefig('./figs/adversarial/advert_%s_to_%s' % (label[0], target_num))
        # plt.show()


if __name__ == '__main__':
    tweak_matrix()
