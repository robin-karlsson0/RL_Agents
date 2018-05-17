import numpy as np
import tensorflow as tf
import random
import os


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
