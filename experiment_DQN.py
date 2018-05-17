import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

#from agent_DQN import Qnetwork_DQN

from utils import ExpBuffer
from utils import createTargetNetworkUpdateOperations
from utils import updateTargetNetwork
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
Qnetwork_update_frequency = int(1e3)
gamma = 0.99
learning_rate = int(1e-4)
save_ep_interval = int(250e3)
pre_train_steps = int(0)
random_action_chance_initial = 1.0
random_action_chance_final   = 0.1
random_action_chance_annealing_steps = int(1e6/4)

save_path = "./saves"
save_model = False

load_model = False

feature_vector_length = 100800 # Atari : 100800
image_h = 210
image_w = 160
image_c = 3

# Initialize environment
env = gym.make(env_name)
printEnvironmentInformation(env, env_name)
action_num = env.action_space.n

# Initialize experience buffer 
exp_buffer = ExpBuffer(buffer_size=exp_buffer_size)

# Exploration
random_action_chance = random_action_chance_initial
rancom_action_chance_drop_per_step = (random_action_chance_initial - random_action_chance_final) / random_action_chance_annealing_steps


#########################
#  COMPUTATIONAL GRAPH  #
#########################

# QNet_moving

# CNN function approximator for processing input image (state 's') into a Q values for each action.
# 1. Input image as a column feature vectors
QNet_moving_input = tf.placeholder(shape=[None, feature_vector_length], dtype=tf.float32, name="input")
# 2. Reshape column feature vectors into an image three-channeled image arrays
QNet_moving_input_image = tf.reshape(QNet_moving_input, shape=[-1, image_h, image_w, image_c], name="input_image")
# 3. Convolution layer 1
QNet_moving_conv1 = slim.conv2d(inputs=QNet_moving_input_image,
                                 num_outputs=32,
                                 kernel_size=[8,8],
                                 stride=[4,4],
                                 padding="VALID",
                                 biases_initializer=None)
# 4. Convolution layer 2
QNet_moving_conv2 = slim.conv2d(inputs=QNet_moving_conv1,
                                 num_outputs=64,
                                 kernel_size=[4,4],
                                 stride=[2,2],
                                 padding="VALID",
                                 biases_initializer=None)
# 5. Convolution layer 3
QNet_moving_conv3 = slim.conv2d(inputs=QNet_moving_conv2,
                                 num_outputs=64,
                                 kernel_size=[3,3],
                                 stride=[1,1],
                                 padding="VALID",
                                 biases_initializer=None)
# 6. Flatten convolution layers into single column vectors
QNet_moving_flat3 = slim.flatten(QNet_moving_conv3)
# 7. Fully connected layers
QNet_moving_full4 = slim.fully_connected(QNet_moving_flat3, 512, activation_fn=tf.nn.relu, biases_initializer=None)
# 8. Linear activation function to output Q-values for each action
QNet_moving_Q_array = slim.fully_connected(QNet_moving_full4, action_num, activation_fn=None, biases_initializer=None)
# Get predicted action index or Q value of action
QNet_moving_Q_max_action = tf.argmax(QNet_moving_Q_array, 1)
QNet_moving_Q_max_value = tf.reduce_max(QNet_moving_Q_array, 1, keepdims=True)
# Output the Q-value corresponding to taking action 'a' for every row (i.e. experience sample)
# 1. tf.one_hot    : Create a one-hot row vector for 'a'
# 2. tf.multiply   : Multiply the array of all Q-values with the action one-hot vector
#                    -> Only selected Q-value becomes non-zero
# 3. tf.reduce_sum : Sum all Q-values, resulting in the selected non-zero Q-value
QNet_moving_a_input = tf.placeholder(shape=[None], dtype=tf.int32, name="a_input")
QNet_moving_Q_selected = tf.reduce_sum(tf.multiply(QNet_moving_Q_array, tf.one_hot(QNet_moving_a_input, action_num, dtype=tf.float32)), 1)
# Placeholder for the static target term
QNet_moving_target_term = tf.placeholder(shape=[None,1], dtype=tf.float32)
# Loss function
QNet_moving_loss = tf.reduce_mean( tf.square(QNet_moving_target_term - QNet_moving_Q_selected) )
tf.summary.histogram("QNet_moving_loss", QNet_moving_loss)
# Training model
QNet_moving_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
QNet_moving_updateModel = QNet_moving_trainer.minimize(QNet_moving_loss)


# QNet_target

# CNN function approximator for processing input image (state 's') into a Q values for each action.
# 1. Input image as a column feature vectors
QNet_target_input = tf.placeholder(shape=[None, feature_vector_length], dtype=tf.float32, name="input")
# 2. Reshape column feature vectors into an image three-channeled image arrays
QNet_target_input_image = tf.reshape(QNet_target_input, shape=[-1, image_h, image_w, image_c], name="input_image")
# 3. Convolution layer 1
QNet_target_conv1 = slim.conv2d(inputs=QNet_target_input_image,
                                num_outputs=32,
                                kernel_size=[8,8],
                                stride=[4,4],
                                padding="VALID",
                                biases_initializer=None)
# 4. Convolution layer 2
QNet_target_conv2 = slim.conv2d(inputs=QNet_target_conv1,
                                num_outputs=64,
                                kernel_size=[4,4],
                                stride=[2,2],
                                padding="VALID",
                                biases_initializer=None)
# 5. Convolution layer 3
QNet_target_conv3 = slim.conv2d(inputs=QNet_target_conv2,
                                num_outputs=64,
                                kernel_size=[3,3],
                                stride=[1,1],
                                padding="VALID",
                                biases_initializer=None)
# 6. Flatten convolution layers into single column vectors
QNet_target_flat3 = slim.flatten(QNet_target_conv3)
# 7. Fully connected layers
QNet_target_full4 = slim.fully_connected(QNet_target_flat3, 512, activation_fn=tf.nn.relu, biases_initializer=None)
# 8. Linear activation function to output Q-values for each action
QNet_target_Q_array = slim.fully_connected(QNet_target_full4, action_num, activation_fn=None, biases_initializer=None)
# Get predicted action index or Q value of action
QNet_target_Q_max_action = tf.argmax(QNet_target_Q_array, 1)
QNet_target_Q_max_value = tf.reduce_max(QNet_target_Q_array, 1, keepdims=True)
# Output the Q-value corresponding to taking action 'a' for every row (i.e. experience sample)
# 1. tf.one_hot    : Create a one-hot row vector for 'a'
# 2. tf.multiply   : Multiply the array of all Q-values with the action one-hot vector
#                    -> Only selected Q-value becomes non-zero
# 3. tf.reduce_sum : Sum all Q-values, resulting in the selected non-zero Q-value
QNet_target_a_input = tf.placeholder(shape=[None], dtype=tf.int32, name="a_input")
QNet_target_Q_selected = tf.reduce_sum(tf.multiply(QNet_target_Q_array, tf.one_hot(QNet_target_a_input, action_num, dtype=tf.float32)), 1)
# Placeholder for the static target term
QNet_target_target_term = tf.placeholder(shape=[None,1], dtype=tf.float32)
# Loss function
QNet_target_loss = tf.reduce_mean( tf.square(QNet_target_target_term - QNet_target_Q_selected) )
# Training model
QNet_target_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
QNet_target_updateModel = QNet_target_trainer.minimize(QNet_target_loss)


# Initialize TensorFlow objects
init = tf.global_variables_initializer()
tf_saver = tf.train.Saver()
tf_trainable_vars = tf.trainable_variables()
# List with operations to overwrite the 'target' Q-network variables
QNet_target_update_op_list = createTargetNetworkUpdateOperations(tf_trainable_vars)


with tf.Session() as sess:

    sess.run(init)

    # Load model

    train_writer = tf.summary.FileWriter( './logs/1/train', sess.graph)

    step_tot = 0
    for ep in range(ep_max):

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
                a = sess.run(QNet_moving_Q_max_action, feed_dict={QNet_moving_input:s})[0]

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
            Q_target_max_value = sess.run(QNet_target_Q_max_value, feed_dict={QNet_target_input:s})
            # Chance target to 'r' in case episode is terminated
            ep_termination = 1 - minibatch_d

            target_term = minibatch_r + gamma * Q_target_max_value * ep_termination

            
            #################
            #  TRAIN MODEL  #
            #################

            #merge = tf.summary.merge_all()

            _ = sess.run(QNet_moving_updateModel, feed_dict={QNet_moving_target_term:target_term,
                                                             QNet_moving_input:minibatch_s_next,
                                                             QNet_moving_a_input:minibatch_a})
            #summary, _ = sess.run([merge, QNet_moving_updateModel], feed_dict={QNet_moving_target_term:target_term,
            #                                                 QNet_moving_input:minibatch_s_next,
            #                                                 QNet_moving_a_input:minibatch_a})

            #train_writer.add_summary(summary, step_tot)


            # 5. Overwite 'target' with 'moving' Q-network
            if(step_tot % Qnetwork_update_frequency == 0):
                merge = tf.summary.merge_all()
                summary = sess.run(merge, feed_dict={QNet_moving_target_term:target_term,
                                                             QNet_moving_input:minibatch_s_next,
                                                             QNet_moving_a_input:minibatch_a})
                train_writer.add_summary(summary, step_tot)
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
