# -*- coding: utf-8 -*-
"""
MNIST example

@author: Gaoruiqi
"""
import tensorflow as tf
import DeepLearning as dl
from tensorflow.examples.tutorials.mnist import input_data
#import data set
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
#reset tensorboard graph
tf.reset_default_graph()
#placeholder
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
#data pre process
x_image = tf.reshape(xs,[-1,28,28,1])
#main graph
with tf.name_scope('conv1') as scope:
    conv1 = dl.cnn(x_image).conv(5,1,32,1,padding='SAME')
with tf.name_scope('pool1') as scope:
    pool1 = dl.cnn(conv1).pooling(2,2,method='max',padding='SAME')
with tf.name_scope('conv2') as scope:
    conv2 = dl.cnn(pool1).conv(5,32,64,1,padding='SAME')
with tf.name_scope('pool2') as scope:
    pool2 = dl.cnn(conv2).pooling(2,2,method='max',padding='SAME')
with tf.name_scope('fc1') as scope:
    fc1 = dl.cnn(pool2).FC(7*7*64,1024,isFlatten=True)
with tf.name_scope('drop') as scope:
    drop = dl.cnn(fc1).dropOut(keep_prob)
with tf.name_scope('prediction') as scope:
    prediction = dl.cnn(drop).FC(1024,10,isSoftmax=True)
with tf.name_scope('cross_entory') as scope:
    cross_entropy = dl.train.crossEntropy(prediction,ys)
    tf.summary.scalar('cross_entropy',cross_entropy)
#train
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
merged=tf.summary.merge_all()
#run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('/graphs',sess.graph)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i % 50 == 0:
            print(dl.train.accuracy(sess,prediction,xs,ys,mnist.test.images,mnist.test.labels,keep_prob,isKeepProb=True))
            summary = sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
            writer.add_summary(summary,i)
