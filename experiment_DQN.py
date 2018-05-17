import numpy as np
import tensorflow as tf
import os

from agent_DQN import Qnetwork_DQN
from agent_DQN import ExpBuffer
from agent_DQN import createTargetNetworkUpdateOperations
from agent_DQN import updateTargetNetwork

from utils import setAllRandomSeeds
from utils import toFeatureVector
from utils import saveModel
from utils import printEnvironmentInformation


import gym


tf.reset_default_graph()


# Set random seeds
setAllRandomSeeds(seed=42)


# Experiment parameters
env_name = "Pong-v0"
ep_max = int(1e9)
step_max = int(1e6)
exp_buffer_size = int(1e6)
minibatch_size = int(32)
Qnetwork_update_frequency = 4
gamma = 0.99
learning_rate = int(1e-4)
save_ep_interval = int(250e3)
pre_train_steps = int(0)
random_action_chance_initial = 1.0
random_action_chance_final   = 0.1
random_action_chance_annealing_steps = int(1e6)

save_path = "./saves"
save_model = True

load_model = False

feature_vector_length = 100800 # Atari : 100800


# Initialize environment
env = gym.make(env_name)
printEnvironmentInformation(env, env_name)
action_num = env.action_space.n


# Initialize Q networks
#   QN_moving : Updated every timestep
#   QN_target : Is only updated at an interval
QNet_moving = Qnetwork_DQN(feature_vector_length, action_num, eta=learning_rate, image_h=210, image_w=160, image_c=3)
QNet_target = Qnetwork_DQN(feature_vector_length, action_num, eta=learning_rate, image_h=210, image_w=160, image_c=3)


########################
#  TENSORFLOW OBJECTS  #  
########################
init = tf.global_variables_initializer()
tf_saver = tf.train.Saver()
tf_trainable_vars = tf.trainable_variables()
# List with operations to overwrite the 'target' Q-network variables
QNet_target_update_op_list = createTargetNetworkUpdateOperations(tf_trainable_vars)

# Initialize experience buffer 
exp_buffer = ExpBuffer(buffer_size=exp_buffer_size)

# Exploration
random_action_chance = random_action_chance_initial
rancom_action_chance_drop_per_step = (random_action_chance_initial - random_action_chance_final) / random_action_chance_annealing_steps


with tf.Session() as sess:

    sess.run(init)

    # Load model

    step_tot = 0
    for ep in range(ep_max):

        #if(step_tot % 1 == 0):
        

        # Reset environment at the start of a new episode
        s    = env.reset()
        s    = toFeatureVector(s)
        done = False

        # Sum of rewards collected during an episode
        ep_r = 0

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

            #env.render()

            # 1. Choose an action 'a' based on higest 'Q' being in 's' (predict) or randomly (explore)
            if(np.random.rand(1) < random_action_chance or step_tot < pre_train_steps):
                a = env.action_space.sample()
            else:
                a = sess.run(QNet_moving.Q_max_action, feed_dict={QNet_moving.input:s})[0]

            # 2. Take action and receive 's_next, r, d' from environment
            s_next, r, done, _ = env.step(a)
            s_next = toFeatureVector(s_next)

            # Add reward to episode total reward
            ep_r += r

            # 3. Store experience to buffer
            exp = (s,a,r,s_next,done)
            exp_buffer.store(exp)

            # Reduce exploration
            if(step_tot > pre_train_steps):
                if(random_action_chance > random_action_chance_final):
                    random_action_chance -= rancom_action_chance_drop_per_step
                    if(random_action_chance < random_action_chance_final):
                        random_action_chance = random_action_chance_final
            
            # 4. Train Q-network if current step is an 'update step'
            '''Training algorithm:
               1. Sample a minibatch of randomly sampled experiences.
                  - 2D numpy array where each column correspond to a stored experience.
                  - np.vstack() turns the "array of arrays" into an "array of primitive elements"
                      shape : (5,) -> (5,100800)
               2. 
            '''
            minibatch = exp_buffer.sample(minibatch_size)

            minibatch_a      = minibatch[:,1]
            minibatch_r      = np.vstack(minibatch[:,2])
            minibatch_s_next = np.vstack(minibatch[:,3])
            minibatch_d      = np.vstack(minibatch[:,4])


            #################
            #  TARGET TERM  #
            #################
            # Maximum Q-value from the target Q-network
            Q_target_max_value = sess.run(QNet_target.Q_max_value, feed_dict={QNet_target.input:s})
            # Chance target to 'r' in case episode is terminated
            ep_termination = 1 - minibatch_d

            target_term = minibatch_r + gamma * Q_target_max_value * ep_termination

            
            #################
            #  TRAIN MODEL  #
            #################
            _ = sess.run(QNet_moving.updateModel, feed_dict={QNet_moving.target_term:target_term,
                                                             QNet_moving.input:minibatch_s_next,
                                                             QNet_moving.a_input:minibatch_a})


            # 5. Overwite 'target' with 'moving' Q-network
            if(step_tot % Qnetwork_update_frequency == 0):
                updateTargetNetwork(QNet_target_update_op_list, sess)              


            S = s_next


            if done:
                print("ep {:d} | steps {:d} | ep_r {:.0f} | exp {:.2f}".format(ep,step_tot,ep_r,random_action_chance))
                break

        # Save
        if(ep % save_ep_interval == 0):
            saveModel(ep, save_path, tf_saver, sess, save_model)


    # Final save
    saveModel(ep, save_path, tf_saver, sess, save_model)
