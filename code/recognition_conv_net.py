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


def iterate_networks(filepath, dropout, regular_factor, niter):
    images,labels = samples(filepath = filepath)
    print ('Data loaded!')



    def next_batch(batch_dim, images, labels):
        perm = np.arange(labels.shape[0])
        np.random.shuffle(perm)
        im = images[perm]
        lab = labels[perm]
        start = 0
        batch_im = im[start:batch_dim]
        trans_batch = np.zeros_like(batch_im)
        for i in range (batch_dim):
            this_image = batch_im[i]
            this_image = this_image.reshape([96,96])
            this_image = get_transformed_ims(this_image)
            trans_batch[i] = this_image.flatten()
        return trans_batch, lab[start:batch_dim]

    def oneHot(labels, n_classes):
        label_oneHot = np.zeros([labels.shape[0], n_classes])
        for j in range(labels.shape[0]):
            label_oneHot[j,labels[j]] = 1
        return label_oneHot




    # Split the database in 75% training set and 25% test set
    im_tr_D = dict()
    im_test_D = dict()
    lab_tr_D = dict()
    lab_test_D = dict()

    im_tr_to_concatenate = []
    im_test_to_concatenate = []
    lab_tr_to_concatenate = []
    lab_test_to_concatenate = []
    for i in range(0, 8):
        loc_im = images[np.where(labels == i)[0], :]
        loc_lab = labels[np.where(labels == i)[0], :]
        loc_im, loc_lab = next_batch(len(loc_im), loc_im, loc_lab)

        if len(loc_im)>4:
            im_tr_D[i] = loc_im[0:int(len(loc_im) * 0.75)]
            lab_tr_D[i] = loc_lab[0:int(len(loc_im) * 0.75)]
            im_test_D[i] = loc_im[int(len(loc_im) * 0.75):]
            lab_test_D[i] = loc_lab[int(len(loc_im) * 0.75):]
            im_tr_to_concatenate.append(im_tr_D[i])
            im_test_to_concatenate.append(im_test_D[i])
            lab_tr_to_concatenate.append(lab_tr_D[i])
            lab_test_to_concatenate.append(lab_test_D[i])

    im_tr = np.concatenate(im_tr_to_concatenate, axis=0)
    im_test = np.concatenate(im_test_to_concatenate, axis=0)
    lab_tr = np.concatenate(lab_tr_to_concatenate, axis=0)
    lab_test = np.concatenate(lab_test_to_concatenate, axis=0)

    print(im_tr.shape, im_test.shape, lab_tr.shape, lab_test.shape)

    # Old method to split database into training set and test set
    #
    # im,lab = next_batch(labels.shape[0],images,labels)
    # #Training set
    # im_tr = im[0:int(labels.shape[0]*0.75)]
    # lab_tr = lab[0:int(labels.shape[0]*0.75)]
    # #Test set
    # im_test = im[int(labels.shape[0]*0.75),:]
    # lab_test = lab[int(labels.shape[0]*0.75),:]

    # Parameters
    learning_rate = 0.001
    training_iters = niter
    batch_size = 200
    display_step = 10

    image_size = 96 # for a squared image, number of pixels (choosing only dimension divisible for two)
    channels_number= 1 #if RGB channels_number = 3, 1 if gray
    kernel_dimension = 3
    number_pooling_layers = 5
    image_dim_fin =  image_size/(2**number_pooling_layers)  #     image dimension after all pooling layers


    # Network Parameters
    n_input = image_size*image_size #  data input

    n_classes = 8 # total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to turn off units   -- it prevents overfitting

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    #reg_factor = tf.placeholder(tf.float32, name='regularization_factor')
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    is_train =  tf.Variable(True,  name='training')
    reg_factor = tf.Variable(regular_factor)

    #one hot encoding
    lab_tr_OH = oneHot(lab_tr, n_classes)
    lab_test_OH = oneHot(lab_test, n_classes)


    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias, no relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        return tf.nn.bias_add(x, b)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
        padding='SAME')




    def batch_norm(x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps (number of filters)
            phase_train: boolean tf.Varialbe, true indicates training phase
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


    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, image_size, image_size, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Batch normalization
        conv1 = batch_norm(conv1, 16, is_train)
        # RELU
        conv1 = tf.nn.relu(conv1)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Batch normalization
        conv2 = batch_norm(conv2, 64, is_train)
        # RELU
        conv2 = tf.nn.relu(conv2)
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Convolution Layer
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        # Batch normalization
        conv3 = batch_norm(conv3, 64, is_train)
        # RELU
        conv3 = tf.nn.relu(conv3)
        # Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=2)

        # Convolution Layer
        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
        # Batch normalization
        conv4 = batch_norm(conv4, 64, is_train)
        # RELU
        conv4 = tf.nn.relu(conv4)
        # Max Pooling (down-sampling)
        conv4 = maxpool2d(conv4, k=2)

        # Convolution Layer
        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
        # Batch normalization
        conv5 = batch_norm(conv5, 64, is_train)
        # RELU
        conv5 = tf.nn.relu(conv5)
        # Max Pooling (down-sampling)
        conv5 = maxpool2d(conv5, k=2)


        # Fully connected layer
        # Reshape conv5 output to fit fully connected layer input
        fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])


        return out


    #Store layers weight & bias
    weights = {
        # 3x3 (kernel dimension), 1 input (1 channel gray image), 16 outputs (number of filters)
        'wc1': tf.Variable(tf.random_normal([kernel_dimension, kernel_dimension, channels_number, 16])),
        # 3x3 conv, 16 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([kernel_dimension, kernel_dimension, 16, 64])),
        # 3x3 conv, 64 inputs, 64 outputs
        'wc3': tf.Variable(tf.random_normal([kernel_dimension, kernel_dimension, 64, 64])),
        # 3x3 conv, 64 inputs, 64 outputs
        'wc4': tf.Variable(tf.random_normal([kernel_dimension, kernel_dimension, 64, 64])),
        # 3x3 conv, 64 inputs, 64 outputs
        'wc5': tf.Variable(tf.random_normal([kernel_dimension, kernel_dimension, 64, 64])),
        # fully connected, 3*3*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([image_dim_fin*image_dim_fin*64, 1024])),   #weights of the first fully-connected layer
        # 1024 inputs, 8 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))  #weights of the last fully-connected layer
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([16])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bc4': tf.Variable(tf.random_normal([64])),
        'bc5': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # List of weights
    weight_list = []
    for i in weights.keys():
        weight_list.append(weights[i])
    for j in biases.keys():
        weight_list.append(biases[j])


    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)



    # Define loss and optimizer
    cost_plain = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    l2_norm = 0.
    for w in weights.keys():
       l2_norm = tf.add(l2_norm, tf.nn.l2_loss(weights[w]))


    cost_regularization = tf.scalar_mul(reg_factor, l2_norm)

    cost = tf.add(cost_plain, cost_regularization)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=weight_list)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()


    # Launch the graph
    with tf.Session() as sess:



        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            #batch_x, batch_y = database.train.next_batch(batch_size)
            batch_x, batch_y = next_batch(batch_size,im_tr,lab_tr_OH)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                print("\n\nTesting Accuracy:", \
                      sess.run(accuracy, feed_dict={x: im_test,
                                                    y: lab_test_OH,
                                                    keep_prob: 1.}))

            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for n test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: im_test,
                                          y: lab_test_OH,
                                          keep_prob: 1.}))
        a  = sess.run(tf.argmax(pred,1), feed_dict={x: im_test,
                                          y: lab_test_OH,
                                          keep_prob: 1.})

        acc_test = sess.run(accuracy, feed_dict={x: im_test,
                                          y: lab_test_OH,
                                          keep_prob: 1.})


    conf_matrix = np.zeros([n_classes, n_classes])
    for i in range(len(a)):
        conf_matrix[lab_test[i],a[i]] +=1

    return conf_matrix, acc_test, acc


if __name__ == '__main__':
    pass