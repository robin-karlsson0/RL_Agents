import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import random
import gym

import matplotlib.pyplot as plt

from collections import deque

from utils import preprocessImage



tf.reset_default_graph()

#####################
#  HYPERPARAMETERS  #
#####################
env_name = "PongNoFrameskip-v4"
ep_max = int(1e6)          # max number of episodes to learn from
step_tot_max = int(200e6)
step_max = int(1e6)                # max steps in an episode
gamma = 0.99                   # future reward discount
learning_rate = float(1e-4)         # Q-network learning rate
seed = 42
action_repeat = 4
# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.1            # minimum exploration probability
explore_steps = int(1e6)
explore_decay_rate = (explore_start - explore_stop) / explore_steps            # exponential decay rate for exploration prob
# Network parameters
frame_buffer_size = 4
hidden_size = 512               # number of units in each Q-network hidden layer
QNet_update_freq = int(1e4)
# Experience buffer parameters
exp_buffer_size = int(1e6)            # ExpBuffer capacity
minibatch_size = 32                # experience mini-batch size
pretrain_step_max = minibatch_size   # number experiences to pretrain the ExpBuffer
# Save parameters
model_save_interval = 100
model_save_dir = "./saves"
save_model = False
# TensorBoard
tensorboard_update_freq = 50


# Initialize environment
env = gym.make(env_name)
env_actions = env.action_space.n
# Set random seeds
def setAllRandomSeeds(environment, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    environment.seed(seed)
setAllRandomSeeds(env, seed)


print("\n*********************")
print("*  Run information  *")
print("*********************")
print("  Environment name   : {}".format(env_name))
print("  Action space       : {}".format(env.action_space))
print("  Observation space  : {}".format(env.observation_space))
print("  ep_max             : {}".format(ep_max))
print("  step_tot_max       : {}".format(step_tot_max))
print("  step_max           : {}".format(step_max))
print("  gamma              : {}".format(gamma))
print("  learning_rate      : {}".format(learning_rate))
print("  seed               : {}".format(seed))
print("  action_repeat      : {}".format(action_repeat))
print("  explore_start      : {}".format(explore_start))
print("  explore_stop       : {}".format(explore_stop))
print("  explore_steps      : {}".format(explore_steps))
print("  explore_decay_rate : {}".format(explore_decay_rate))
print("  frame_buffer_size  : {}".format(frame_buffer_size))
print("  QNet_update_freq   : {}".format(QNet_update_freq))
print("  pretrain_step_max  : {}".format(pretrain_step_max))
print("  exp_buffer_size    : {}".format(exp_buffer_size))
print("  minibatch_size     : {}".format(minibatch_size))
print("")
print("  model_save_interval : {}".format(model_save_interval))
print("  model_save_dir      : {}".format(model_save_dir))
print("  save_model          : {}".format(save_model))
print("")
print("  tensorboard_update_freq : {}".format(tensorboard_update_freq))
print("")

##############################
#  EXPERIENCE REPLAY BUFFER  #
##############################
class ExpBuffer():
    '''Class for storing and sampling experiences.

    An experience 'exp' is represented as a list entry [s, a, r, s_next, d]:
      s [float] : 1D array representing the state.
      a (int)   : Action taken during the step.
      r (float) : Reward attained during the step.
      s_next [float] : 1D array representing the resulting state having taken action 'a'
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
          tuple of arrays
            s : [s1, s2, s3, ...]
            a : [a1, a2, a3, ...]
            ...
        '''
        # Sample array of integers representing indices of experiences in the buffer
        index_array = np.random.choice(len(self.buffer), size=minibatch_size, replace=False)
        # Add sampled experiences to a list
        exp_list = [self.buffer[i] for i in index_array]
        # Parse list elements into arrays
        s      = np.array([exp[0] for exp in exp_list])
        a      = np.array([exp[1] for exp in exp_list])
        r      = np.array([exp[2] for exp in exp_list])
        s_next = np.array([exp[3] for exp in exp_list])
        d      = np.array([exp[4] for exp in exp_list])
        return (s, a, r, s_next,d)

ExpBuffer = ExpBuffer(max_size=exp_buffer_size)


##################
#  FRAME BUFFER  #
##################
class FrameBuffer():
    '''
    '''
    def __init__(self, size = 4):
        self.buffer = deque(maxlen=size)

    def store(self, frame):
        # Add frame to the back, pushing out the frame in the front
        # [1,2,3,4].append(0) -> [2,3,4,0]
        self.buffer.append(frame)

    def sample(self):
        # Return all buffered frames as a 3D array
        frame_1 = self.buffer[0]
        frame_2 = self.buffer[1]
        frame_3 = self.buffer[2]
        frame_4 = self.buffer[3]
        frame_stack = np.stack((frame_1, frame_2, frame_3, frame_4), axis=2)
        return frame_stack

    def reset(self, env):
        for i in range(frame_buffer_size):
            s = env.reset()
            s = preprocessImage(s)
            self.buffer.append(s)

buff_s = FrameBuffer(size=4)
buff_s_next = FrameBuffer(size=4)


#########################
#  COMPUTATIONAL GRAPH  #
#########################\
def createTargetNetworkUpdateOperations(tf_trainable_vars):
    '''Create a list where each element is a tensorflow assign operation which overwrites
       trainable variables of 'target' with 'moving'.

    First HALF of all variables correpsond to the 'moving' Q-network.
    Later HALF are the 'target' Q-network.
    [ ... moving ... | ... target ... ]

    Note: The Q-network objects must be initialized in THIS ORDER !!!

    Args:
      tf_trainable_vars (tf var list) : List containing all trainable variables.
    Returns:
      op_list (list) : List containing tf operations to copy 'moving' -> 'target'.
    '''
    tot_vars = len(tf_trainable_vars)
    op_list = []
    # Get the 'moving' Q-network variables from the first half ('//'' is integer division)
    for i, moving_var in enumerate(tf_trainable_vars[0:tot_vars//2]):
        # Define a tf operation which copies the 'moving' variable into 'target' variable
        op_list.append( tf_trainable_vars[i+tot_vars//2].assign(moving_var.value()) )

    return op_list


def updateTargetNetwork(op_list, sess):
    '''Run previously defined tf operations to overwrite 'target' Q-network with 'moving' Q-network.'''
    for op in op_list:
        sess.run(op)


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


class QNetwork:
    '''Class which represent the network which outputs a Q-value based on an input state.
    '''
    def __init__(self, learning_rate, img_w, img_h, frame_num, action_size, hidden_size, name='QNetwork'):
        '''Initialize all tensorflow operation objects when initializing Q-network object.

        Use the CNN function approximation to generate and store Q-value arrays.
        These Q-value arrays are then inputted into 'Q_input' placeholder in order to compute other things.
        '''
        # state inputs to the Q-network
        with tf.variable_scope(name):

            ##################
            #  PLACEHOLDERS  #
            ##################
            # Array with an integer for every sample
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            # One hot encode the actions to later choose the Q-value for the action
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            ###########
            #  INPUT  #
            ###########
            self.inputs = tf.placeholder(tf.float32, [None, img_w, img_h, frame_num], name='inputs')
            
            ############
            #  LAYERS  #
            ############

            self.conv1 = slim.conv2d(inputs=self.inputs,
                                     num_outputs=32,
                                     kernel_size=[8,8],
                                     stride=[4,4],
                                     padding="VALID",
                                     biases_initializer=None)

            self.conv2 = slim.conv2d(inputs=self.conv1,
                                     num_outputs=64,
                                     kernel_size=[4,4],
                                     stride=[2,2],
                                     padding="VALID",
                                     biases_initializer=None)

            self.conv3 = slim.conv2d(inputs=self.conv2,
                                     num_outputs=64,
                                     kernel_size=[3,3],
                                     stride=[1,1],
                                     padding="VALID",
                                     biases_initializer=None)

            self.flatten = slim.flatten(self.conv3)

            self.fc4 = slim.fully_connected(self.flatten, hidden_size, activation_fn=tf.nn.relu, biases_initializer=None)

            self.output = slim.fully_connected(self.fc4, action_size, activation_fn=None, biases_initializer=None)

            # ReLU hidden layers
            #self.fc1 = tf.contrib.layers.fully_connected(self.inputs, hidden_size)
            #self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            # Linear output layer
            #self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, activation_fn=None)
            
            ##############
            #  TRAINING  #
            ##############
            # Value of selected Q-value          
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            # L2 loss (targetQ - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            tf.summary.scalar("loss", self.loss)
            # Training operations
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            # TensorBoard operation
            self.merged = tf.summary.merge_all()

QNet_moving = QNetwork(learning_rate, 84, 84, 4, env_actions, hidden_size, name='QNet_moving')
QNet_target = QNetwork(learning_rate, 84, 84, 4, env_actions, hidden_size, name='QNet_target')


tf_init = tf.global_variables_initializer()
tf_saver = tf.train.Saver()
tf_trainable_vars = tf.trainable_variables()
# List with operations to overwrite the 'target' Q-network variables
QNet_target_update_op_list = createTargetNetworkUpdateOperations(tf_trainable_vars)


############################
#  INITIALIZE ENVIRONMENT  #
############################
# Fill frame buffers
buff_s.reset(env)
buff_s_next.reset(env)


####################
#  PRETRAIN STEPS  #
####################
# Fill replay buffer with randomly selected experiences before doing training steps
for pretrain_step in range(pretrain_step_max):
    # Make a random action
    a = env.action_space.sample()
    s_next, r, done, _ = env.step(a)
    s_next = preprocessImage(s_next)
    buff_s_next.store(s_next)
    # Store experience to buffer
    ExpBuffer.store((buff_s.sample(), a, r, buff_s_next.sample(), done))
    if done:      
        # Start new episode
        env.reset()
        # Reset buffers
        buff_s.reset(env)
        buff_s_next.reset(env)
        s, r, done, _ = env.step(env.action_space.sample())
        s = preprocessImage(s)
        buff_s.store(s)
        buff_s_next.store(s)

    else:
        s = s_next
        buff_s.store(s)


###############
#  MAIN LOOP  #
###############
saver = tf.train.Saver()
rewards_list = []
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf_init)
    updateTargetNetwork(QNet_target_update_op_list, sess)
    # TensorBoard storage directory
    tb_train_writer = tf.summary.FileWriter('./logs/1/train', sess.graph)

    step_tot = 0

    for ep in range(1, ep_max):

        # Reset buffer at start of episode
        buff_s.reset(env)
        buff_s_next.reset(env)

        # Reset environment at start of episode
        s = env.reset()
        s = preprocessImage(s)
        buff_s.store(s)
        buff_s_next.store(s_next)
        done = False
        step = 0
        # Store total reward for the episode
        ep_rew = 0       

        while step < step_max:

            step_tot += 1
            # env.render() 
            
            ###################
            #  SELECT ACTION  #
            ###################
            # Reduce exploration
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-explore_decay_rate*step_tot) 
            if explore_p > np.random.rand():
                # Select exploratory random action
                a = env.action_space.sample()
            else:
                # Select optimal action according to policy (i.e. Q-network)
                Qs = sess.run(QNet_moving.output, feed_dict={QNet_moving.inputs: np.reshape(buff_s.sample(), (1,84,84,4))})
                a = np.argmax(Qs)
            
            #################
            #  TAKE ACTION  #
            #################
            # Do 'action repeat' -> Train only on every 4th frame
            r_repeat_tot = 0
            for _ in range(action_repeat):
                s_next, r_repeat, done, _ = env.step(a)
                r_repeat_tot += r_repeat
                if(done == True):
                    break
            r = r_repeat_tot
            
            s_next = preprocessImage(s_next)
            buff_s_next.store(s_next)

            ######################
            #  STORE EXPERIENCE  #
            ######################
            ExpBuffer.store((buff_s.sample(), a, r, buff_s_next.sample(), done))             
            
            ##############
            #  TRAINING  #
            ##############
            # Sample one minibatch from buffer into separate arrays
            # Each array index correspond to the same experience
            s_minibatch, a_minibatch, r_minibatch, s_next_minibatch, done_minibatch = ExpBuffer.sample(minibatch_size)

            #print("s_minibatch.shape = {}".format(s_minibatch.shape))
            #print("s_next_minibatch.shape = {}".format(s_next_minibatch.shape))
            #input("")
            
            # Compute 'target term'
            ep_termination = 1 - done_minibatch
            target_Q_array = sess.run(QNet_target.output, feed_dict={QNet_target.inputs: s_next_minibatch})
            targets = r_minibatch + gamma * np.max(target_Q_array, axis=1)*ep_termination

            # Update moving Q-network
            _ = sess.run(QNet_moving.opt, feed_dict={QNet_moving.inputs: s_minibatch,
                                                     QNet_moving.targetQs_: targets,
                                                     QNet_moving.actions_: a_minibatch})

            ###########################
            #  UPDATE TARGET NETWORK  #
            ###########################
            if(step_tot % QNet_update_freq == 0):
                updateTargetNetwork(QNet_target_update_op_list, sess) 

            # Update TensorBoard output
            if(step_tot % tensorboard_update_freq == 0):
                tb_summary = sess.run(QNet_moving.merged, feed_dict={QNet_moving.inputs: s_minibatch,
                                                                     QNet_moving.targetQs_: targets,
                                                                     QNet_moving.actions_: a_minibatch})
                tb_train_writer.add_summary(tb_summary, step_tot)

            # End of step
            s = s_next
            buff_s.store(s)
            step += 1
            ep_rew += r

            # End episode
            if(done == True):
                print('Episode: {}'.format(ep),
                      'Steps : {}'.format(step_tot),
                      'Total reward: {}'.format(ep_rew),
                      'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep,ep_rew,step_tot))
                break

    
        # Save model checkpoint
        if(ep % model_save_interval == 0):
            saveModel(ep, model_save_dir, tf_saver, sess, save_model)



eps, rews, steps = np.array(rewards_list).T
eps = np.reshape(eps, [-1,1])
rews = np.reshape(rews, [-1,1])
steps = np.reshape(steps, [-1,1])
out_array = np.concatenate((steps, rews),axis=1)
print(out_array.shape)
# Output rewards per episode
np.savetxt("rew_new.dat", out_array, fmt="%d")
#plt.plot(rewards_list)
#plt.show()

env.close()
