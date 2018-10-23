import numpy as np
import random as rand
import gym
import matplotlib.pyplot as plt
import replay_fn as rpf
import keras_neural_network as qnet
import tensorflow as tf
from tensorflow import keras
import dqn_agent

# Set a bunch of parameters

epsilon            = 1.0
epsilon_decay      = 0.993
gamma              = 0.99
steps_per_episode  = 1000
target_update_freq = 200
mem_dimension      = 100000
n_minibatches      = 64
save_model         = 1000


# Load trained weights from file

num_episodes             = 1000
my_dqn_agent             = dqn_agent(gamma, steps_per_episode, target_update_freq, mem_dimension, n_minibatches, save_model)
trained_network          = keras.models.load_model('learning_network.h5', custom_objects=None, compile=True)
trained_network_weights  = trained_network.get_weights()

test_rewards, test_run_lengths = my_dqn_agent.run_trained_model(trained_network_weights, num_episodes,  record_video=True)

print("Mean Reward Over 100 Episodes: ", np.mean(np.array(test_rewards)), "+/-", np.std(np.array(test_rewards)) )
print("Mean Total Episodes Counts: ", np.mean(np.array(test_run_lengths)), "+/-", np.std(np.array(test_run_lengths)) )

