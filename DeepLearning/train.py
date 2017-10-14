"""
Deep Learning Class
@author: Gaoruiqi
"""
import tensorflow as tf
class train:
    @staticmethod
    def accuracy(session,outputLayer,xBatch,yBatch,data,label,keep_prob,isKeepProb=False):
        """
        params
            session : session that is runing
            outputLayer : the prediction layer
            xBatch : feed_dict associated with data
            yBatch : feed_dict associated with label
            data : input data
            label : lable of input data
            keep_prob : dropout ratio
            isKeepProb : if there is a dropout parameter or not
        return
            accuracy of prediction
        """
        if isKeepProb:
            y_pre = session.run(outputLayer, feed_dict={xBatch:data,keep_prob:1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(label,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            result = accuracy.eval({xBatch:data,yBatch:label,keep_prob:1})
        else:
            y_pre = session.run(outputLayer, feed_dict={xBatch:data})
            correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(label,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            result = accuracy.eval({xBatch:data,yBatch:label})
        return result
    @staticmethod
    def crossEntropy(prediction,label):
        """
        params
            predict : input data
            label : lable of input data
        return
            cross entropy
        """
        crossEntropy = tf.reduce_mean(-tf.reduce_sum(label*tf.log(prediction),reduction_indices=[1]))
        return crossEntropy
    @staticmethod
    def squareError(prediction,label):
        """
        params
            predict : input data
               label : lable of input data
        return
            square error
        """
        squareError = tf.reduce_mean(tf.reduce_sum(tf.square(label-prediction),reduction_indices=[1]))
        return squareError
