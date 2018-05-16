import numpy as np
import tensorflow as tf

from agent_DQN import Qnetwork_DQN
from agent_DQN import ExpBuffer

from utils import setAllRandomSeeds
from utils import printEnvironmentInformation
from utils import toFeatureVector

import gym

tf.reset_default_graph()

# Set random seeds
setAllRandomSeeds(seed=21)


# Experiment parameters
ep_max = 5
step_max = 10
env_name = "Pong-v0"
exp_buffer_size = 1e6
minibatch_size = 32
Qnetwork_update_frequency = 4
gamma = 0.99

feature_vector_length = 100800

# Initialize environment
env = gym.make(env_name)
printEnvironmentInformation(env, env_name)
action_num = env.action_space.n

# Initialize Q networks
#   QN_moving : Updated every timestep
#   QN_target : Is only updated at an interval
QNet_moving = Qnetwork_DQN(feature_vector_length, action_num)
QNet_target = Qnetwork_DQN(feature_vector_length, action_num)

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

            #Q = sess.run(QNet_moving.Q_array, feed_dict={QNet_moving.input:s})
            #print(Q)

            



            # 4. Train Q-network if current step is an 'update step'
            if(step_tot % Qnetwork_update_frequency == 0):
                '''Training algorithm:
                   1. Sample a minibatch of randomly sampled experiences.
                      - 2D numpy array where each column correspond to a stored experience.
                      - np.vstack() turns the "array of arrays" into an "array of primitive elements"
                          shape : (5,) -> (5,100800)
                   2. 
                '''
                minibatch = exp_buffer.sample(minibatch_size)

                #print(minibatch[:,3])
                #print(minibatch[:,3].shape)
                #print(np.vstack(minibatch[:,3]))
                #print(np.vstack(minibatch[:,3]).shape)
                #input("Knut")

                #Q_moving = sess.run(QNet_moving.Q_array, feed_dict={QNet_moving.input:np.vstack(minibatch[:,3])})

                minibatch_a      = minibatch[:,1]
                minibatch_r      = np.vstack(minibatch[:,2])
                minibatch_s_next = np.vstack(minibatch[:,3])
                minibatch_d      = np.vstack(minibatch[:,4])

                #################
                #  TARGET TERM  #
                #################

                Q_target = sess.run(QNet_target.Q_array, feed_dict={QNet_target.input:minibatch_s_next})

                # Maximum Q-value from the target Q-network
                Q_target_max_value = sess.run(QNet_target.Q_max_value, feed_dict={QNet_target.Q_input:Q_target})
                # Chance target to 'r' in case episode is terminated
                ep_termination = 1 - minibatch_d

                target_term = minibatch_r + gamma * Q_target_max_value * ep_termination


                #################
                #  MOVING TERM  #
                #################
                
                # Q-value of selected action 'a'
                #moving_term = sess.run(QNet_moving.Q_selected, feed_dict={QNet_moving.Q_input:Q_moving, QNet_moving.a_input:minibatch_a})
                #moving_term = np.vstack(moving_term)

                
                #################
                #  TRAIN MODEL  #
                #################

                _ = sess.run(QNet_moving.updateModel, feed_dict={QNet_moving.target_term:target_term,
                                                                 QNet_moving.input:minibatch_s_next,
                                                                 QNet_moving.a_input:minibatch_a})



                


                


            # ...

            S = s_next

            if done:
                break


