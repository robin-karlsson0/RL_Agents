import numpy as np
import tensorflow as tf
import random
import os


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


def setAllRandomSeeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    

def toFeatureVector(array):
    '''Flatten input 2D array into a 1D feature vector.
    '''
    return np.reshape(array, [1,-1])


def saveModel(ep, save_dir, tf_saver, sess, save_model=False):
    '''Create a save model folder within the specified save directory.

       Args:
         ep (int)             : Episode number where the model is saved.
         save_dir (str)       : Path to the directory where the model will be saved
         tf_saver (tf object) : 
         sess : (tf object)   : 
         save_model : (bool)  : 
    '''
    if(save_model == False):
        return
    # Create the save directory if it does not already exist
    if(os.path.isdir(save_dir) == False):
        os.mkdir(save_dir)
    # Create save folder
    save_folder_path = os.path.join(save_dir, "save_{}".format(ep))
    os.mkdir(save_folder_path)
    # Define save name
    file_name = "save_{}.ckpt".format(ep)
    save_path = os.path.join(save_folder_path, file_name)

    print("Saving model : {}".format(save_path))

    tf_saver.save(sess, save_path)


def printEnvironmentInformation(env, env_name):
    print("")
    print("*****************************")
    print("*  Environment information  *")
    print("*****************************")
    print("  Environment name  : {}".format(env_name))
    print("  Action space      : {}".format(env.action_space))
    print("  Observation space : {}".format(env.observation_space))
    print("")
