import numpy as np
import tensorflow as tf
import random

def setAllRandomSeeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    

def toFeatureVector(array):
    '''Flatten input 2D array into a 1D feature vector.
    '''
    return np.reshape(array, [1,-1])


def printEnvironmentInformation(env, env_name):
    print("")
    print("*****************************")
    print("*  Environment information  *")
    print("*****************************")
    print("  Environment name  : {}".format(env_name))
    print("  Action space      : {}".format(env.action_space))
    print("  Observation space : {}".format(env.observation_space))
    print("")