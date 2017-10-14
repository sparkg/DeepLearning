# -*- coding: utf-8 -*-
"""
Deep Learning Class

@author: Gaoruiqi
"""
import tensorflow as tf

class cnn:
    def __init__(self,
                 x,
                 layerName='layer',
                 convName='conv',
                 poolName='pool',
                 FCName='FC'
                ):
        """
        params
            x : input data
        return
            None
        """
        self.inputData=x#input data is 4dims [batch,in_height,in_width,in_channels]
        self.convName=layerName+convName
        self.poolName=layerName+poolName
        self.FCName=layerName+FCName
    def conv(self,ksize,in_c,knums,stride,padding='SAME',symmetric=True):
        """
        params
            ksize : convolution kernel size,a number when kernel is symmetric,a list when kernel is asymmetric
            in_c : input channels number
            knums : kernel number
            stride : filter stride
            symmetric : is kernel symmetric
        return
            convolution feature map
        """
        assert ((type(in_c) is int)and(type(knums) is int)and(type(stride) is int)and(type(padding) is str)),\
        "input data type(s) is(are) wrong"
        if symmetric:
            assert(type(ksize) is int),"type of kernel size is wrong"
            kernel=tf.Variable(tf.truncated_normal([ksize,ksize,in_c,knums],stddev=0.1),name='kernel')
        elif not symmetric:
            assert ((type(ksize) is list) and (len(ksize) == 2)),"type or length of kernel size is wrong"
            kernel=tf.Variable(tf.truncated_normal([ksize[0],ksize[1],in_c,knums],stddev=0.1),name='kernel')
        else:
            print("cannot determine if kernel is symmetric,treat it as default")
            kernel=tf.Variable(tf.truncated_normal([ksize,ksize,in_c,knums],stddev=0.1),name='kernel')
        bias=tf.Variable(tf.constant(0.1,shape=[knums]),name='bias')
        conv=tf.nn.conv2d(self.inputData,kernel,strides=[1,stride,stride,1],padding=padding,name=self.convName)
        activation=tf.nn.relu(conv+bias)
        return activation
    def pooling(self,ksize,stride,method='max',padding='SAME'):
        """
        params
            ksize : pool kernel size
            stride : filter stride
            method : pooling method
            padding : padding method
        return
            pooling feature map
        """
        assert ((type(ksize) is int)and(type(stride) is int)and(type(method) is str)and(type(padding) is str)),\
        "input data type(s) is(are) wrong"
        if method == 'average':
            pool = tf.nn.avg_pool(self.inputData,[1,ksize,ksize,1],[1,stride,stride,1],padding=padding,name=self.poolName)
        elif method == 'max':
            pool = tf.nn.max_pool(self.inputData,[1,ksize,ksize,1],[1,stride,stride,1],padding=padding,name=self.poolName)
        else:
            print('Wrong pooling method,use max_pool instead')
            pool = tf.nn.max_pool(self.inputData,[1,ksize,ksize,1],[1,stride,stride,1],padding=padding,name=self.poolName)
        return pool
    def FC(self,in_num,out_num,isFlatten=False,isSoftmax=False):
        """
        params
            in_num : input number of full connection layer
            out_num : output number of full connection layer
            isFlatten : if input data need to be flatten
            isSoftmax : is output activation is softmax
        return
            full connection layer
        """
        assert ((type(in_num) is int)and(type(out_num) is int)),\
        "input data type(s) is(are) wrong"
        if isFlatten:
            self.inputData=tf.reshape(self.inputData,[-1,in_num])
        else:
            pass
        weight =tf.Variable(tf.truncated_normal([in_num,out_num],stddev=0.1),name='weight')
        bias = tf.Variable(tf.constant(0.1,shape=[out_num]),name='bias')
        if isSoftmax:
            output = tf.nn.softmax(tf.matmul(self.inputData,weight)+bias,name=self.FCName)
        else:
            output = tf.nn.relu(tf.matmul(self.inputData,weight)+bias,name=self.FCName)
        return output
    def dropOut(self,keep_prob=1):
        """
        params
            keep_prob : dropout ratio
        return
            dropout layer
        """
        assert ((type(keep_prob) is int)or(type(keep_prob) is float))or(keep_prob.dtype == (tf.float32 or tf.float16 or tf.float64)),"input data type(s) is(are) wrong"
        dropout = tf.nn.dropout(self.inputData,keep_prob)
        return dropout
