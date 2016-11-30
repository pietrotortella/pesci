# -*- coding: UTF-8 -*-

#Convolution neural network for recognition copied from whale competition (head localizer architecture)

import numpy as np
import random
import tensorflow as tf
import math
import cv2
from skimage.transform import rotate
from colors_noise import saltpepper
import matplotlib.pyplot as plt
#from ... import fishdatabase   #import data from fishdatabase

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from tensorflow.models.image.cifar10 import cifar10
from create_database import samples

filepath = '/home/terminale2/Documents/ALL_small.json'


def get_transformed_ims(im):
    MAX_TRANS = 0.35
    MAX_SP = 0.08
    MAX_TILT = 12

    trans_im = np.array(im)
    sp_perc = np.random.rand(1)[0] * MAX_SP
    trans_im, _ = saltpepper(trans_im, sp_perc)

    tilt_degree = (np.random.rand(1)[0] - 0.5) * 2 * MAX_TILT
    trans_im = rotate(trans_im, tilt_degree)

    trans_len = np.random.rand(1)[0] * MAX_TRANS
    trans_theta = np.random.rand(1)[0] * 360
    trans_theta = math.radians(trans_theta)
    trans_v = (int(trans_len * im.shape[0] * math.sin(trans_theta)),
               int(trans_len * im.shape[1] * math.cos(trans_theta)))

    trans2_im = np.zeros_like(trans_im)

    temp = np.array(trans_im[
                    max(0, -trans_v[0]):im.shape[0] + min(0, -trans_v[0]),
                    max(0, -trans_v[1]):im.shape[1] + min(0, -trans_v[1])
                    ])

    trans2_im[
    max(0, trans_v[0]):im.shape[0] + min(0, trans_v[0]),
    max(0, trans_v[1]):im.shape[1] + min(0, trans_v[1])
    ] = temp

    ims = trans2_im

    return ims


# def next_batch(batch_dim, images, labels, batch_no, random_seed=123):
#     perm = np.arange(labels.shape[0])
#     rng = np.random.RandomState(random_seed)
#     rng.shuffle(perm)
#     im = images[perm]
#     lab = labels[perm]
#     start = batch_no * batch_dim
#     stop = (batch_no + 1) * batch_dim
#     batch_im = im[start:stop]
#     trans_batch = np.zeros_like(batch_im)
#     for i in range(batch_dim):
#         this_image = batch_im[i]
#         this_image = this_image.reshape([96, 96])
#         this_image = get_transformed_ims(this_image)
#         trans_batch[i] = this_image.flatten()
#     return trans_batch, lab[start:stop]


def next_batch(batch_dim, x, y, batch_no, random_seed=123,
               noise_data_function=None):
    perm = np.arange(x.shape[0])
    rng = np.random.RandomState(random_seed)
    rng.shuffle(perm)
    start = batch_no * batch_dim
    stop = (batch_no + 1) * batch_dim
    batch_x = x[perm[start:stop]]
    if noise_data_function is not None:
        batch_x = noise_data_function(batch_x)
    batch_y = y[perm[start:stop]]

    return batch_x, batch_y


def oneHot(labels, n_classes):
    label_oneHot = np.zeros([labels.shape[0], n_classes])
    for j in range(labels.shape[0]):
        label_oneHot[j, labels[j]] = 1
    return label_oneHot


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps (number of filters)
        phase_train: boolean tf.Variable, true indicates training phase
    Return:
        normed:      batch-normalized maps
    """

    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed


class ConvLayer():
    def __init__(self, in_tensor, out_channels, phase_train,
                 kernel_dimension=3, strides=1,
                 padding='SAME', perform_batch_norm=True,
                 activation=tf.nn.relu):

        self.in_tensor = in_tensor
        in_channels = self.in_tensor.get_shape()[-1].value
        # in_channels = tf.shape(self.in_tensor)[-1]

        W_val = np.array(
            np.random.randn(kernel_dimension, kernel_dimension,
                            in_channels, out_channels),
            dtype=np.float32
        )
        # W_shape = (kernel_dimension, kernel_dimension,
        #                     in_channels, out_channels)

        self.W = tf.Variable(
            W_val,
            # tf.random_normal(
            #     shape=W_shape,
            #     dtype=tf.float32,
            #     name='W_initial_val'
            # ),
            dtype=tf.float32,
            name='W'
        )
        b_val = np.array(
            np.random.randn(out_channels),
            dtype=np.float32,
        )
        self.b = tf.Variable(
            b_val,
            # tf.random_normal(
            #     [out_channels]
            # ),
            dtype=tf.float32,
            name='b'
        )

        pre_output = tf.nn.conv2d(self.in_tensor, self.W,
                                  strides=[1, strides, strides, 1],
                                  padding=padding)

        if perform_batch_norm:
            pre_output = batch_norm(pre_output, out_channels, phase_train)

        self.out_tensor = activation(tf.nn.bias_add(pre_output, self.b))


class PoolLayer():
    def __init__(self, in_tensor, k=2, padding='SAME'):

        self.in_tensor = in_tensor
        self.out_tensor = tf.nn.max_pool(self.in_tensor, ksize=[1, k, k, 1],
                                         strides=[1, k, k, 1],
                                         padding=padding)


class DenseLayer():
    def __init__(self, in_tensor, out_dim, dropout_keep=1, activation=tf.nn.relu):
        self.in_tensor = in_tensor
        in_dim = self.in_tensor.get_shape()[-1]

        W_val = np.array(
            np.random.randn(in_dim, out_dim),
            dtype=np.float32
        )
        self.W = tf.Variable(
            W_val,
            dtype=tf.float32,
            name='W'
        )
        b_val = np.array(
            np.random.randn(out_dim),
            dtype=np.float32,
        )
        self.b = tf.Variable(
            b_val,
            dtype=tf.float32,
            name='b'
        )
        pre_output = tf.add(tf.matmul(self.in_tensor, self.W), self.b)
        pre_output = activation(pre_output)
        self.out_tensor = tf.nn.dropout(pre_output, dropout_keep)


class ConvNN():
    """
    Remark that after a DENSE layer one can have only DENSE layers,
    and POOL layers are allowed only immediately after a CONV layer.

    samples of the dictionaries to feed in

    CONVOLUTIONAL LAYER:
    d_sample_conv = {
        'type': 'conv'
        'n_out': int, the number of channels in the output
        'kernel_dim': int, the dimension of the kernel
            (at the moment it supports only square kernels)
        'strides': int, the amount of the stride
        'padding': string, the type of padding (e.g. 'SAME')
        'perform_batch_norm': bool, True if one wants to perform
            batch normalization
        'activation': a tensorflow function (e.g. tf.nn.relu)
    }

    MAX POOLING LAYER
    d_sample_pool = {
        'type': 'pool'
        'k': int, the size of the pooling window
        'padding': string, the type of padding (e.g. 'SAME')
    }

    DENSE LAYER
    d_sample_dense = {
        'type': 'dense'
        'n_out': int, the number of outputs (e.g. the number of
            neurons in the next layer)
        'dropout_keep': float, between 0 and 1, the keep probability
            of the droupout step (set = 1 to have no dropout)
        'activation': a tensorflow function (e.g. tf.nn.relu)
    }
    """
    def __init__(self, n_initial_features, layer_dictlist,
                 image_y_size=None, image_x_size=None, n_channels=None):

        # to keep track if at the moment we are in a training
        self.in_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[None, n_initial_features],
            name='X'
        )
        # phase or not, mostly for the batch_normalization layers
        self.phase_train = tf.placeholder(dtype=bool, name='is_train')
        # stores the layers of the nn as pairs (type, layer)
        self.layer = []
        # stores the parameters of the nn, i.e. the weights one
        # modifies during the training
        self.params = []
        # keeps track of the last output of the feedforward
        last_out = self.in_tensor
        # keeps track of the dimension of the last Dense hidden layer
        # to have the dimension of the output of the nn
        last_out_dim = -1

        # the main loop where we initialize the layers
        count_conv = 0
        count_pool = 0
        count_dense = 0
        for d in layer_dictlist:
            if d['type'] == 'conv':
                if len(last_out.get_shape()) != 4:
                    if not (image_y_size is not None and
                            image_x_size is not None and
                            n_channels is not None):
                        raise NotImplementedError('shapes of the image not provided!')
                    in_tensor = tf.reshape(last_out, shape=[-1, image_y_size, image_x_size, n_channels])
                else:
                    in_tensor = last_out
                with tf.name_scope('conv_' + str(count_conv + 1)) as scope:
                    this_layer = ConvLayer(
                        in_tensor=in_tensor,
                        out_channels=d['n_out'],
                        phase_train=self.phase_train,
                        kernel_dimension=d['kernel_dim'],
                        strides=d['strides'],
                        padding=d['padding'],
                        perform_batch_norm=d['perform_batch_norm'],
                        activation=d['activation']
                    )
                last_out = this_layer.out_tensor
                self.layer.append(('conv', this_layer))
                self.params.append(this_layer.W)
                self.params.append(this_layer.b)
                count_conv += 1

            elif d['type'] == 'pool':
                if self.layer[-1][0] != 'conv':
                    raise NotImplementedError('POOL layer should follow a CONV layer!')
                this_layer = PoolLayer(
                    in_tensor=last_out,
                    k=d['k'],
                    padding=d['padding']
                )
                last_out = this_layer.out_tensor
                self.layer.append(('pool', this_layer))
                count_pool += 1

            elif d['type'] == 'dense':
                if len(last_out.get_shape()) != 2:
                    first_shape = -1 # last_out.get_shape()[0].value
                    second_shape = np.array(
                        [last_out.get_shape()[i].value
                         for i in range(1, len(last_out.get_shape()))]
                    ).prod()
                    new_shape = (first_shape, second_shape)

                    in_tensor = tf.reshape(
                        last_out,
                        shape=new_shape
                    )
                else:
                    in_tensor = last_out

                with tf.name_scope('dense_' + str(count_dense + 1)) as scope:
                    this_layer = DenseLayer(
                        in_tensor=in_tensor,
                        out_dim=d['n_out'],
                        dropout_keep=d['dropout_keep'],
                        activation=d['activation']
                    )
                last_out = this_layer.out_tensor
                last_out_dim = d['n_out']
                self.layer.append(('dense', this_layer))
                self.params.append(this_layer.W)
                self.params.append(this_layer.b)
                count_dense += 1

        self.n_conv = count_conv
        self.n_pool = count_pool
        self.n_dense = count_dense

        # the final output of the nn
        self.out_tensor = last_out
        self.out_dim = last_out_dim
        self.y_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.out_dim],
            name='y'
        )

        self.plain_loss = None
        self.reg_loss = None
        self.train_loss = None
        self.optimizer = None
        self.train = None
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')
        self.reg_l1 = tf.placeholder(dtype=tf.float32, name='reg_l1')
        self.reg_l2 = tf.placeholder(dtype=tf.float32, name='reg_l2')

        # initialize to None attributes for later uses
        self.sess = None
        self.eval_train_funs = []
        self.eval_train_funs_names = []

    def compile(self,
                metric=tf.nn.softmax_cross_entropy_with_logits,
                eval_train_funs=[]):
        self.plain_loss = tf.reduce_mean(metric(self.out_tensor, self.y_tensor))
        self.reg_loss = self.reg_l1 * 1 + self.reg_l2 * 1
        self.train_loss = tf.add(self.plain_loss, self.reg_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train = self.optimizer.minimize(self.train_loss, var_list=self.params)

        for fun_name, fun in eval_train_funs:
            self.eval_train_funs_names.append(fun_name)
            self.eval_train_funs.append(fun(self.out_tensor, self.y_tensor))

    def open_session(self):
        init = tf.initialize_all_variables()
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(init)

    def close_session(self):
        if self.sess:
            self.sess.close()
            self.sess = None

    def fit(self, X, y, valid_set=None,
            n_epochs=10, batch_size=100,
            lr=0.01, reg_l1=0, reg_l2=0,
            noise_data_function=None,
            verbose=1,
            interactive_plot=False):

        train_cost_history = []
        train_loss_history = []
        test_loss_history = []
        for epoch in range(n_epochs):
            batch_costs = []
            for j in range(X.shape[0] // batch_size):
                batch_x, batch_y = next_batch(batch_size, X, y,
                                              batch_no=j,
                                              random_seed=epoch,
                                              noise_data_function=noise_data_function)
                _, batch_cost = self.sess.run(
                    [self.train, self.train_loss],
                    feed_dict={
                        self.in_tensor: batch_x,
                        self.y_tensor: batch_y,
                        self.lr: lr,
                        self.phase_train: True,
                        self.reg_l1: reg_l1,
                        self.reg_l2: reg_l2
                    }
                )
                batch_costs.append(batch_cost)
            val_train_cost = np.array(batch_costs).mean()
            train_cost_history.append(val_train_cost)

            if verbose:
                if epoch % verbose == 0:
                    val_train_loss = self.sess.run(
                        self.plain_loss,
                        feed_dict={
                            self.in_tensor: X,
                            self.y_tensor: y,
                            self.phase_train: False
                        }
                    )
                    train_loss_history.append(val_train_loss)
                    if valid_set is not None:
                        val_test_loss = self.sess.run(
                            self.plain_loss,
                            feed_dict={
                                self.in_tensor: valid_set[0],
                                self.y_tensor: valid_set[1],
                                self.phase_train: False
                            }
                        )
                        test_loss_history.append(val_test_loss)
                        valid_msg = 'Test cost: {:>8.5f}'.format(val_test_loss)
                    else:
                        valid_msg = ''
                    print(
                        'Epoch # {:>5}. Train cost: {:>8.5f}. Train loss: {:>8.5f}'
                        ''.format(epoch, val_train_cost, val_train_loss) + valid_msg
                    )

            if interactive_plot:
                import matplotlib.pyplot as plt

                plt.ion()
                plt.subplot(211)
                plt.axis([0, 100, 0, 2])
                plt.plot(list(range(epoch+1)), train_cost_history, color='pink')
                plt.plot(list(range(epoch + 1)), [t ** 2 for t in train_cost_history], color='black')
                plt.legend()
                plt.subplot(212)
                plt.axis([0, 100, 0, 1])
                plt.plot(list(range(epoch+1)), train_cost_history, color='purple')
                plt.pause(0.005)

        return train_cost_history, train_loss_history, test_loss_history

    def predict(self, X, y):
        prediction = self.sess.run(
            self.out_tensor,
            feed_dict={
                self.in_tensor: X,
                self.phase_train: False
            }
        )
        return prediction


if __name__ == '__main__':
    pass