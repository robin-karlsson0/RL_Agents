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
    exp_buffer = ExpBuffer()

    # Run experiment
    for ep in range(ep_max):

        s = env.reset()
        s = toFeatureVector(s)

        d = False

        for step in range(step_max):

            env.render()

            #Q = sess.run(QN_moving.Q_out, feed_dict={QN_moving.input:s})
            Q = sess.run(QN_target.Q_out, feed_dict={QN_target.input:s})
            
            # Decide on what action to take
            action = env.action_space.sample()

            # Perform action in environment
            s_new, rew, done, _ = env.step(action)
            s_new = toFeatureVector(s_new)



            # ...

            S = s_new

            if done:
                break

