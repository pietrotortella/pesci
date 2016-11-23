from __future__ import print_function
import tensorflow as tf
import numpy as np
from random import shuffle
import copy
import pickle

######################################################################
with open('train/train/ANNOTATION/fishList.pkl', 'rb') as input:
    fishList = pickle.load(input)
#shuffling the list for 10 times
print('Shuffling the list for 10 times...')
for i in range(10):
    shuffle(fishList)

print ('fish list length: ', len(fishList))
testPercentage = int((80*len(fishList))/100)
#80% set for training
fishListTrain = copy.copy(fishList[:testPercentage])
print ('Train set length: ',len(fishListTrain))
#20% set for training
fishListTest = copy.copy(fishList[testPercentage:])
print ('Test set length: ',len(fishListTest))
# Parameters

learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 1

# Network Parameters
n_input = 65536 # (img shape: 256*256)
n_classes = 12 # total out puts - x, y, hight, width, tailx, taily, headx, heady
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
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
    conv3 = maxpool2d(conv3, k=2)

    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)

    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
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
    # 3x3 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    # 3x3 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # 3x3 conv, 64 input, 64 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # fully connected, 8*8*64 inputs (becuse of two pooling layer), 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
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
        shuffle(fishListTrain)
        batch_x = np.empty((0, 65536), int)
        batch_y = np.empty((0, 12), int)
        for i in range(batch_size):
            batch_x = np.append(batch_x, np.array([fishListTrain[i].fishPixels]), axis=0)
            oneY = [fishListTrain[i].fish_X, fishListTrain[i].fish_Y, fishListTrain[i].fish_H, fishListTrain[i].fish_W, fishListTrain[i].head_X, fishListTrain[i].head_Y, fishListTrain[i].tail_X, fishListTrain[i].tail_Y, fishListTrain[i].upfin_X, fishListTrain[i].upfin_Y, fishListTrain[i].lowfin_X, fishListTrain[i].lowfin_Y]
            batch_y = np.append(batch_y, np.array([oneY]), axis=0)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 20% test images
    test_x = np.empty((0, 65536), int)
    test_y = np.empty((0, 8), int)
    for test in fishListTest:
        test_x = np.append(test_x, np.array([test.fishPixels]), axis=0)
        oneY = [test.fish_X, test.fish_Y, test.fish_H, test.fish_W, test.head_X, test.head_Y, test.tail_X, test.tail_Y, test.upfin_X, test.upfin_Y, test.lowfin_X, test.lowfin_Y]
        test_y = np.append(test_y, np.array([oneY]), axis=0)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_x,
                                      y: test_y,
                                      keep_prob: 1.}))
