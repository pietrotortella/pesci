'''
A Convolutional Network implementation example using TensorFlow library.
This network construction is based on and example in github
https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/3_NeuralNetworks/convolutional_network
This network has 5 layers and the structure is copied from the right whale competition  https://deepsense.io/deep-learning-right-whale-recognition-kaggle/
It is the structure of the head localizer.
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''




from __future__ import print_function

import json

from DetectionDatabase_from_json import getTarget
from DetectionDatabase_from_json import getData

import numpy as np

import tensorflow as tf


json_path = "/home/terminale8/Documents/pesci/annotations/TIN_DOL.json"
mypath = "/home/terminale8/Documents/Kaggle_project/train"
jsn_file = open(json_path).read()
jsn_data = json.loads(jsn_file)

#import data and target values
data = getData(jsn_data)
target=getTarget(jsn_data)



# Parameters
learning_rate = 0.001
training_iters = 100
batch_size = 10
display_step = 1

# Network Parameters
n_input = 256*256 #  data input (img shape: 256*256)
n_parameters = 16 # number of parameters in each image
dropout = 0.75 # Dropout, probability to keep units (avoid overfitting)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_parameters])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 256, 256, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv3, k=2)

    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv4, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    # reshape to make matrix multiplicatio prossiblek
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 16 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 16])),
    # 3x3 conv, 1 input, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 16, 64])),
    # 3x3 conv, 1 input, 64 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # 3x3 conv, 1 input, 64 outputs
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # 3x3 conv, 1 input, 64 outputs
    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # fully connected, 8*8*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 16 outputs (parameter prediction)
    'out': tf.Variable(tf.random_normal([1024, n_parameters]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([64])),



    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_parameters]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
        batch_x, batch_y = data[(step-1)*batch_size:step*batch_size], target[(step-1)*batch_size:step*batch_size]
        print(np.shape(batch_x), np.shape(batch_y))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.}) #dropout apply only in training set to aoid overfitting (regularization)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

# Test part still need to be implemented

#     # Calculate accuracy for 256 mnist test images
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
#                                       y: mnist.test.labels[:256],
# keep_prob: 1.}))