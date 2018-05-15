import numpy as np
import tensorflow as tf

from agent_DQN import Qnetwork_DQN
from agent_DQN import ExpBuffer

from utils import printEnvironmentInformation
from utils import toFeatureVector

import gym

# TensorFlow
tf.reset_default_graph()
tf.set_random_seed(22)

# Experiment parameters
ep_max = 5
step_max = 10
env_name = "Pong-v0"
exp_buffer_size = 1e6
minibatch_size = 32
Qnetwork_update_frequency = 4

feature_vector_length = 100800

# Initialize environment
env = gym.make(env_name)
printEnvironmentInformation(env, env_name)
action_num = env.action_space.n

# Initialize Q networks
#   QN_moving : Updated every timestep
#   QN_target : Is only updated at an interval
QN_moving = Qnetwork_DQN(feature_vector_length, action_num)
QN_target = Qnetwork_DQN(feature_vector_length, action_num)

# Initialize TensorFlow objects
init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)

    # Initialize experience buffer 
    exp_buffer = ExpBuffer(buffer_size=exp_buffer_size)

    # Run experiment
    step_tot = 0
    for ep in range(ep_max):

        # Reset environment at the start of a new episode
        s = env.reset()
        s = toFeatureVector(s)
        d = False

        for step in range(step_max):
            '''Procedure when taking a step:
               0. Be in state 's'
               1. Choose an action 'a' based on higest 'Q' being in 's' (predict)
               2. Take action and receive 's_next, r, d' from environment
               3. Store experience to buffer
               4. If current step is an 'update step':
                  -> Train Q-network on ONE minibatch of randomly sampled experiences
            '''
            step_tot += 1

            print("ep : {}, step : {}".format(ep,step))

            env.render()

            # 1. Choose an action 'a' based on higest 'Q' being in 's' (predict)
            a = env.action_space.sample()

            # 2. Take action and receive 's_next, r, d' from environment
            s_next, r, done, _ = env.step(a)
            s_next = toFeatureVector(s_next)

            # 3. Store experience to buffer
            exp = (s,a,r,s_next,d)
            exp_buffer.store(exp)

            # 4. Train Q-network if current step is an 'update step'
            if(step_tot % Qnetwork_update_frequency == 0):
                '''Training algorithm:
                   1. Create a minibatch of randomly sampled experiences.
                      - 2D numpy array where each column correspond to a stored experience.
                   2. 
                '''
                minibatch = exp_buffer.sample(minibatch_size)

                #Q = sess.run(QN_moving.Q_out, feed_dict={QN_moving.input:s})
                #Q = sess.run(QN_target.Q_out, feed_dict={QN_target.input:s})  


            # ...

            S = s_next

            if done:
                break

print(exp_buffer.sample(2).shape)