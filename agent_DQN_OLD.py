import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

class Qnetwork_DQN():
    '''Class which represent the network which outputs a Q-value based on an input state.
    '''

    def __init__(self, input_size, action_num, eta=0.0001, image_h=210, image_w=160, image_c=3):
        '''Initialize all tensorflow operation objects when initializing Q-network object.

           Use the CNN function approximation to generate and store Q-value arrays.
           These Q-value arrays are then inputted into 'Q_input' placeholder in order to compute other things.

        '''

        # CNN function approximator for processing input image (state 's') into a Q values for each action.
        # 1. Input image as a column feature vectors
        self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="input")
        # 2. Reshape column feature vectors into an image three-channeled image arrays
        self.input_image = tf.reshape(self.input, shape=[-1, image_h, image_w, image_c], name="input_image")
        # 3. Convolution layer 1
        self.conv1 = slim.conv2d(inputs=self.input_image,
                                 num_outputs=32,
                                 kernel_size=[8,8],
                                 stride=[4,4],
                                 padding="VALID",
                                 biases_initializer=None)
        # 4. Convolution layer 2
        self.conv2 = slim.conv2d(inputs=self.conv1,
                                 num_outputs=64,
                                 kernel_size=[4,4],
                                 stride=[2,2],
                                 padding="VALID",
                                 biases_initializer=None)
        # 5. Convolution layer 3
        self.conv3 = slim.conv2d(inputs=self.conv2,
                                 num_outputs=64,
                                 kernel_size=[3,3],
                                 stride=[1,1],
                                 padding="VALID",
                                 biases_initializer=None)
        # 6. Flatten convolution layers into single column vectors
        self.flat3 = slim.flatten(self.conv3)
        # 7. Fully connected layers
        self.full4 = slim.fully_connected(self.flat3, 512, activation_fn=tf.nn.relu, biases_initializer=None)
        # 8. Linear activation function to output Q-values for each action
        self.Q_array = slim.fully_connected(self.full4, action_num, activation_fn=None, biases_initializer=None)

        # Get predicted action index or Q value of action
        self.Q_max_action = tf.argmax(self.Q_array, 1)
        self.Q_max_value = tf.reduce_max(self.Q_array, 1, keepdims=True)


        # Output the Q-value corresponding to taking action 'a' for every row (i.e. experience sample)
        # 1. tf.one_hot    : Create a one-hot row vector for 'a'
        # 2. tf.multiply   : Multiply the array of all Q-values with the action one-hot vector
        #                    -> Only selected Q-value becomes non-zero
        # 3. tf.reduce_sum : Sum all Q-values, resulting in the selected non-zero Q-value
        self.a_input = tf.placeholder(shape=[None], dtype=tf.int32, name="a_input")
        self.Q_selected = tf.reduce_sum(tf.multiply(self.Q_array, tf.one_hot(self.a_input, action_num, dtype=tf.float32)), 1)


        # Placeholder for the static target term
        self.target_term = tf.placeholder(shape=[None,1], dtype=tf.float32)

        # Loss function
        self.loss = tf.reduce_mean( tf.square(self.target_term - self.Q_selected) )
        

        # Training model
        self.trainer = tf.train.AdamOptimizer(learning_rate=eta)
        self.updateModel = self.trainer.minimize(self.loss)

