import numpy as np
import tensorflow as tf

from agent_DQN import DQN

from utils import toFeatureVector

import gym

tf.reset_default_graph()

# Experiment parameters
ep_max = 5
step_max = 10
env_name = "Pong-v0"
feature_vector_length = 100800

# Initialize environment
env = gym.make(env_name)
action_num = env.action_space.n
print(env.observation_space)

# Initialize agent
agent = DQN(feature_vector_length, action_num)

# Initialize TensorFlow objects
init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)

    # Run experiment
    for ep in range(ep_max):

        s = env.reset()
        s = toFeatureVector(s)

        for step in range(step_max):

            env.render()

            Q = sess.run(agent.Q_out, feed_dict={agent.input:s})
            

            action = env.action_space.sample()

            s_new, rew, done, _ = env.step(action)
            s_new = toFeatureVector(s_new)

            # ...

            S = s_new

            if done:
                break

