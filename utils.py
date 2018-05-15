import numpy as np


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