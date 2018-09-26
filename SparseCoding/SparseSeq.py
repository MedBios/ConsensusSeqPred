#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018

@author: mpcr
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import progressbar

#import excel file data#
##some code that does that^

#define some things about the data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

sparsity_constraint = tf.placeholder(tf.float32)

#build the network#
#define variables
with tf.variable_scope('NeuralLayer'):
    W = tf.get_variable('W', shape=[784, 784], initializer=tf.random_normal_initializer(stddev=1e-1))
    b = tf.get_variable('b', shape=[784], initializer=tf.constant_initializer(0.1))

    z = tf.matmul(x, W) + b
    a = tf.nn.relu(z)

    # We graph the average density of neurons activation
    average_density = tf.reduce_mean(tf.reduce_sum(tf.cast((a > 0), tf.float32), axis=[1]))
    tf.summary.scalar('AverageDensity', average_density)
with tf.variable_scope('SoftmaxLayer'):
    W_s = tf.get_variable('W_s', shape=[784, 10], initializer=tf.random_normal_initializer(stddev=1e-1))
    b_s = tf.get_variable('b_s', shape=[10], initializer=tf.constant_initializer(0.1))

    out = tf.matmul(a, W_s) + b_s
    y = tf.nn.softmax(out)
with tf.variable_scope('Loss'):
    epsilon = 1e-7 # After some training, y can be 0 on some classes which lead to NaN 
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y + epsilon), axis=[1]))
    # We add our sparsity constraint on the activations
    loss = cross_entropy + sparsity_constraint * tf.reduce_sum(a)

    tf.summary.scalar('loss', loss) # Graph the loss
summaries = tf.summary.merge_all() # This is convenient

with tf.variable_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.summary.scalar('accuracy', accuracy) 
    
    
#train network
adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = adam.minimize(loss)
sess = None
# We iterate over different sparsity constraint
for sc in [0, 1e-4, 5e-4, 1e-3, 2.7e-3]:
    result_folder = dir + '/results/' + str(int(time.time())) + '-fc-sc' + str(sc)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sw = tf.summary.FileWriter(result_folder, sess.graph)
        
        for i in range(20000):
            batch = mnist.train.next_batch(100)
            current_loss, summary, _ = sess.run([loss, summaries, train_op], feed_dict={
                x: batch[0],
                y_true: batch[1],
                sparsity_constraint: sc
            })
            sw.add_summary(summary, i + 1)

            if (i + 1) % 100 == 0:
                acc, acc_sum = sess.run([accuracy, acc_summary], feed_dict={
                    x: mnist.test.images, 
                    y_true: mnist.test.labels
                })
                sw.add_summary(acc_sum, i + 1)
                print('batch: %d, loss: %f, accuracy: %f' % (i + 1, current_loss, acc))


#compile network
tf.reset_default_graph()
os.system('tensorboard --logdir=' + tb_dir)