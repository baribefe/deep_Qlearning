import gym
import numpy as np

env = gym.make('LunarLander-v2')
for i_episode in range(100):
    observation = env.reset()
    total_reward = []
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward.append(reward)
        if done:
            print("Total Reward: ", np.mean(total_reward)," |", "{} timesteps: ".format(t+1) )
            break
