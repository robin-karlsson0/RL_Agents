import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

class Qnetwork_DQN():
    '''Class which represent the network which outputs a Q-value based on an input state.
    '''

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

       An experience 'exp' is represented as [s, a, r, s_next, d] recorded for a step.
    '''

    def __init__(self, buffer_size=1e6):
        '''Initialize an experience buffer object with an empty buffer and a maximums buffer size.
        '''
        self.buffer = []
        self.buffer_size = buffer_size


    def store(self, exp):
        '''Store a new experience entry in the buffer. Replaces oldest entries with new ones
           if buffer is full.

           Args:
             exp (tuple) : tuple containing an experience entry (s,a,r,s_next,d)
        '''
        if(self.size() == self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append(exp)


    def sample(self, sample_num):
        '''Return a random sample of experience recordings as a 2D numpy array.

           sample_array:
           exp_1 : | s | a | r | s_next | d |
           exp_2 : | s | a | r | s_next | d |
           ...

           1. Create a list of randomly sampled experience elements from the buffer.
           2. Convert the list into a 1D numpy array.
           3. Reshape the 1D numpy array into a 2D numpy array where each row correspond to one experience.

           Args:
             sample_num (int) : number of samples to output.

           Returns:
             sample_array (np 2D array) : each row correspond to one randomly sampled experience recording.
        '''
        sample_array = np.reshape(np.array(random.sample(self.buffer, sample_num)), [sample_num, 5])
        return sample_array


    def size(self):
        '''Return the number of stored experiences within the buffer.
        '''
        return len(self.buffer)