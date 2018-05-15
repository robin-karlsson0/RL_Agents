import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Qnetwork_DQN():

    def __init__(self, input_size, output_size, image_h=210, image_w=160, image_c=3):

        self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32)

        self.input_image = tf.reshape(self.input, shape=[-1, image_h, image_w, image_c])

        self.conv1 = slim.conv2d(inputs=self.input_image,
                           num_outputs=32,
                           kernel_size=[8,8],
                           stride=[4,4],
                           padding="VALID",
                           biases_initializer=None)

        self.conv2 = slim.conv2d(inputs=self.conv1,
                            num_outputs=64,
                            kernel_size=[4,4],
                            stride=[2,2],
                            padding="VALID",
                            biases_initializer=None)

        self.conv3 = slim.conv2d(inputs=self.conv2,
                            num_outputs=64,
                            kernel_size=[3,3],
                            stride=[1,1],
                            padding="VALID",
                            biases_initializer=None)

        self.flat3 = slim.flatten(self.conv3)

        self.full4 = slim.fully_connected(self.flat3, 512, activation_fn=tf.nn.relu, biases_initializer=None)

        self.Q_out = slim.fully_connected(self.full4, output_size, activation_fn=None, biases_initializer=None)



class ExpBuffer():
    '''Class for storing, adding, and sampling experiences.
    '''

    def __init__(self, buffer_size=1e6):

        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, exp):

        if(len(self.buffer) + len(exp) >= self.buffer_size):

            self.buffer[0:(len(exp) + len(self.buffer)) - self.buffer_size] = []

        self.buffer.extend(exp)


    def sample(self, sample_num):

        return np.reshape(np.array(random.sample(self.buffer, sample_num)), [sample_num, 5])