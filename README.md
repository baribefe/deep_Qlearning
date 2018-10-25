# An implementation of Mnih et al deep Q-learning Algorithm to solve the Lunar Lander game hosted on OpenAI Gym (https://gym.openai.com/envs/LunarLander-v2/).

The train_model() function in dqn_agent.py calls the run_one_episode method of the dqn_agent class to train a model with parameters that can be specified. After training, the weights of the neural networks are written to a HDF5 file. The picture below shows total rewards and steps count for a 500 episode training run.

![alt text](https://raw.githubusercontent.com/baribefe/deep_Qlearning/master/learning_plots.png)

A trained model can be loaded from file and used for a test run as shown in the run_trained_dqn_agent.py script.
