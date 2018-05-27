import numpy as np
from collections import deque


class ExpBuffer():
    '''Class for storing and sampling experiences.

    [ [f,a,r,d], [f,a,r,d], [f,a,r,d], ... ]
        exp_0      exp_1      exp_2

    An experience 'exp' is represented as a list entry [f, a, r, d]:
      f [uint8] : 1D array representing a frame.
      a (int)   : Action taken during the step.
      r (float) : Reward attained during the step.
      d (bool) : True if episode was terminated.
    '''
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    
    def store(self, experience):
        # Add exp to the back, pushing out the exp in the front
        self.buffer.append(experience)
            
    def sample(self, minibatch_size):
        '''Return 1D arrays. Each index correspond to a component of the same experience.

        Create a list of length 'minibatch_size' whose elements represents randomly sampled
        experiences from the buffer.

        Returns:
          tuple of arrays (each element correspond to one minibatch sample)
            s : [s1, s2, s3, ...]
            a : [a1, a2, a3, ...]
            ...
        '''
        # Sample array of integers representing indices of experiences in the buffer
        #   NOTE : State needs to be reconstructed from 4 preceding frames!
        buffer_index_array = np.random.choice(np.arange(4,len(self.buffer)), size=minibatch_size, replace=False)

        # Reconstruct states from frames one batch sample at a time
        s = []
        s_next = []
        for buff_i in buffer_index_array:
            # Load frames from preceding experiences
            f_1 = self.buffer[buff_i-4][0]
            f_2 = self.buffer[buff_i-3][0]
            f_3 = self.buffer[buff_i-2][0]
            f_4 = self.buffer[buff_i-1][0]
            f_5 = self.buffer[buff_i-0][0]
            # Stack frames together into a state
            frame_stack = np.stack((f_1, f_2, f_3, f_4), axis=2)
            frame_stack_next = np.stack((f_2, f_3, f_4, f_5), axis=2)
            # Store state for each batch sample in a list
            s.append(frame_stack)
            s_next.append(frame_stack_next)
        # Convert list elements into arrays
        s      = np.array(s)
        s_next = np.array(s_next)

        # Add sampled experiences to a list
        exp_list = [self.buffer[buff_i] for buff_i in buffer_index_array]
        # Parse list elements into arrays
        a = np.array([exp[1] for exp in exp_list])
        r = np.array([exp[2] for exp in exp_list])
        d = np.array([exp[3] for exp in exp_list])

        return (s, a, r, s_next, d)

