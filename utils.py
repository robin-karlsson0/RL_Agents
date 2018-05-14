import numpy as np


def toFeatureVector(array):
    '''Flatten input 2D array into a 1D feature vector.
    '''
    return np.reshape(array, [1,-1])

