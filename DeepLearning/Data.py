import tensorflow as tf
import numpy as np
import os
class Data(object):
   def __init__(self,filepath,label,shuffle=False):
      self.data = np.array(self._read_file_path(filepath))
      self.label = np.array(label)
      self._len_label = len(self.label)
      print("There are {} data and {} labels".format(len(self.data),len(self.label)))
      assert len(self.data) == len(self.label)
      #need shuffle or not
      if shuffle:
         temp = np.arange(len(self.data))
         np.random.shuffle(temp)
         self.data = self.data[temp]
         self.label = self.label[temp]

#read all files path in fileDir
   @staticmethod
   def _read_file_path(fileDir):
      pathname=[]
      for (roots,dirs,files) in os.walk(fileDir):
         for file in files:
            pathname += [os.path.join(roots,file)]
      return pathname
   def next_batch(self,batch_size,height,width,channel,need_preprocess=False):
      #change data categlory to tensor
      self.data = tf.cast(self.data,tf.string)
      self.label = tf.cast(self.label,tf.int32)
      #build queue
      input_queue = tf.train.slice_input_producer([self.data,self.label],shuffle=False)
      #process label
      label = input_queue[1]
      label = tf.reshape(label,[1])
      #process image
      img_content = tf.read_file(input_queue[0])
      image = tf.image.decode_image(img_content,channels=channel)
      #standard
      if need_preprocess:
         image = tf.image.per_image_standardization(image)
      #resize
      image_resize = tf.image.resize_image_with_crop_or_pad(image,height,width)
      image_resize = tf.reshape(image_resize,[height,width,channel])
      #batch
      image_batch,label_batch = tf.train.batch([image_resize,label],batch_size=batch_size,capacity=3*batch_size)
      label_batch = tf.reshape(label_batch,[batch_size])
      return image_batch,label_batch



