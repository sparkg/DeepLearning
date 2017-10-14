"""
Deep Learning Class
@author: Gaoruiqi
"""
import tensorflow
class train:
    @staticmethod
    def accuracy(predict,label,xBatch=xs,yBatch=ys,keep_prob=keep_prob,isKeepProb=False):
        """
        params:predict is input data
               label is lable of input data
               xBatch is feed_dict associated with predict
               yBatch is feed_dict associated with label
               keep_prob is dropout ratio
               isKeepProb shows if there need a dropout parameter or not
        return:accuracy of prediction
        ***run in a tf.Session()
        ***
        """
        global prediction
        if isKeepProb:
            y_pre = sess.run(prediction, feed_dict={xBatch:predict,keep_prob:1})
            correct_prediction = tf.equal(tf.argmax(y_pre, 1),tf.argmax(label,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            result = sess.run(accuracy,feed_dict={xBatch:predict,yBatch:label,keep_prob:1})
        else:
            y_pre = sess.run(prediction, feed_dict={xBatch:predict})
            correct_prediction = tf.equal(tf.argmax(y_pre, 1),tf.argmax(label,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            result = sess.run(accuracy,feed_dict={xBatch:predict,yBatch:label})
        return result
    @staticmethod
    def crossEntropy(prediction,label):
        """
        params:predict is input data
               label is lable of input data
        return:cross entropy
        """
        crossEntropy = tf.reduce_mean(-tf.reduce_sum(label*tf.log(prediction),reduction_indices=[1]))
        return crossEntropy
    @staticmethod
    def squareError(prediction,label):
        """
        params:predict is input data
               label is lable of input data
        return:square error
        """
        squareError = tf.reduce_mean(tf.reduce_sum(tf.square(label-prediction),reduction_indices=[1]))
        return squareError
