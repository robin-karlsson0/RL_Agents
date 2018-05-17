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
        if(sample_num > self.size()):
            sample_num = self.size()
            
        sample_array = np.reshape(np.array(random.sample(self.buffer, sample_num)), [sample_num, 5])
        return sample_array


    def size(self):
        '''Return the number of stored experiences within the buffer.
        '''
        return len(self.buffer)


def createTargetNetworkUpdateOperations(tf_trainable_vars):
    '''Create a list where each element is a tensorflow assign operation which overwrites
       trainable variables of 'target' with 'moving'.

       First HALF of all variables correpsond to the 'moving' Q-network.
       Later HALF are the 'target' Q-network.
       [ ... moving ... | ... target ... ]

       Note: The Q-network objects must be initialized in THIS ORDER !!!

       Args:
         tf_trainable_vars (tf var list) : List containing all trainable variables.
       Returns:
         op_list (list) : List containing tf operations to copy 'moving' -> 'target'.
    '''
    tot_vars = len(tf_trainable_vars)
    op_list = []
    # Get the 'moving' Q-network variables from the first half ('//'' is integer division)
    for i, moving_var in enumerate(tf_trainable_vars[0:tot_vars//2]):
        # Define a tf operation which copies the 'moving' variable into 'target' variable
        op_list.append( tf_trainable_vars[i+tot_vars//2].assign(moving_var.value()) )

    return op_list


def updateTargetNetwork(op_list, sess):
    '''Run previously defined tf operations to overwrite 'target' Q-network with 'moving' Q-network.
    '''
    for op in op_list:
        sess.run(op)
