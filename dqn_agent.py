import numpy as np
import random as rand
import gym
import matplotlib.pyplot as plt
import replay_fn as rpf
import keras_neural_network as qnet
import tensorflow as tf
from tensorflow import keras

class dqn_agent(object):
    def __init__(self, gamma, steps_per_episode, target_update_freq, mem_dimension, n_minibatches, save_model):

        # Initilize environment and learning variables
        self.env                 = gym.make('LunarLander-v2')
        self.num_actions         = 4
        self.gamma               = gamma
        self.steps_per_episode   = steps_per_episode
        self.target_update_freq  = target_update_freq
        self.mem_dimension       = mem_dimension 
        self.n_minibatches       = n_minibatches
        self.save_model          = save_model
        self.total_steps         = 0

        # Initialize replay memory D
        self.D = rpf.reply_fn(mem_size=self.mem_dimension,batch_size=self.n_minibatches)

        # Initialize learning Q neural network using random weights
        self.learn_network = qnet.keras_neural_network(n_classes=4, batch_size=self.n_minibatches, nvariables=8, learning_rate=0.0001) 

        # Initialize target Q neural using the weights of the learning network
        self.target_network = qnet.keras_neural_network(n_classes=4, batch_size=self.n_minibatches, nvariables=8, learning_rate=0.0001)
        self.target_network.copy_weights(self.learn_network.get_weights()) 

    def run_one_episode(self,epsilon,epsilon_decay):
    
        episode_reward = 0
        state = self.env.reset()
        count = 0

        while count <  steps_per_episode:

            self.random_or_optimal = rand.random()
            if self.random_or_optimal <= epsilon: 
                action = rand.randint(0, self.num_actions-1)
            else:
                # Generate action using learning network
                action = np.argmax(self.learn_network.test_neural_network(np.reshape(state, (1,-1))) )

            # Excute action in emulator and observe reward and next state
            s_prime,R, isTerminal, info =  self.env.step(action)
            episode_reward +=  R
 
            # Store rewards in memory D
            self.D.add_to_memory(state,R,action,s_prime,isTerminal)
            state = s_prime
            self.total_steps += 1

            if isTerminal:
                break

            # Sample mini-batches from memory if count is greater than equilibration steps
            memory_samples = self.D.replay_memory(self.n_minibatches)
        
            if memory_samples is not None:
            
                b_states, b_actions, b_Rs, b_s_primes, b_isTerminals = memory_samples

                #  Do greedy policy in this batches
                targets = self.learn_network.test_neural_network(np.array(b_states))
	    
                q_prime = self.target_network.test_neural_network(np.array(b_s_primes))
                q_prime = np.max(q_prime,axis=1)
                q_prime = np.array(b_Rs) + self.gamma*q_prime
                q_prime = np.where(b_isTerminals, 0, q_prime)

                targets[np.arange(len(b_actions)), b_actions] = q_prime            

	        # Do one step of gradient descent
                self.learn_network.train_neural_network(b_states, targets, 1)

            if self.total_steps % self.target_update_freq == 0:
                self.target_network.copy_weights(self.learn_network.get_weights())
            count += 1

        return [count, episode_reward]

    def run_trained_model(self,target_network_weights, num_episodes, record_video=False):

        # Load trained neural network from file
        self.target_network.copy_weights(trained_network_weights)
        self.learn_network.copy_weights(trained_network_weights)
        
        if record_video:
            monitor_env = gym.wrappers.Monitor(self.env, 'recorded_episodes', video_callable=lambda episode_id: episode_id%10==0)

        test_rewards = []
        test_run_lengths = []

        for j in range(num_episodes):

            # Initialize rewards, environment and count
            episode_reward = 0
            state = self.env.reset()
            count = 0

            # Run agent

            while count <  steps_per_episode:

                self.env.render()
                # Generate action using learning network
                action = np.argmax(self.learn_network.test_neural_network(np.reshape(state, (1,-1))) )

                # Excute action in emulator and observe reward and next state
                s_prime,R, isTerminal, info =  self.env.step(action)
                episode_reward +=  R

                state = s_prime
                count += 1

                if isTerminal:
                    break
            test_rewards.append(episode_reward)
            test_run_lengths.append(count)
            print("Reward: ", episode_reward,"|| Count: ", count)

        return [test_rewards, test_run_lengths]

episode_count      = 0
all_rewards        = []
total_epi_counts   = []
epsilon            = 1.0
epsilon_decay      = 0.993
gamma              = 0.99
steps_per_episode  = 1000
target_update_freq = 200
mem_dimension      = 100000
n_minibatches      = 64
save_model         = 500

def train_model():
    my_dqn_agent = dqn_agent(gamma, steps_per_episode, target_update_freq, mem_dimension, n_minibatches, save_model)

    while True:
        one_episode_count, episode_rewards = my_dqn_agent.run_one_episode(epsilon,epsilon_decay)
        print("Episode ", str(episode_count+1)+":", "1 Episode Count: ",one_episode_count , episode_rewards, "epsilon: ", epsilon)
        episode_count += 1
        all_rewards.append(episode_rewards)
        total_epi_counts.append(one_episode_count)
        if epsilon > 0.0001:
            epsilon = epsilon*epsilon_decay
        if episode_count % save_model == 0:
            break 

    #Plot learning
    plt.plot(all_rewards, marker = '.', ms=14) 
    plt.plot(all_rewards, ls = '-', ms=8)
    plt.xlim([0,500])
    plt.xlabel('Training Episodes')
    plt.ylabel('Total Reward')
    plt.title('Deep Q-Learning: Lunar Lander', family='sans-serif',size='18',stretch='ultra-condensed',color='r')
    plt.savefig('rewards.png')
    plt.savefig('rewards.pdf')

    #Save reward sequence to file
    rewardsfile = 'rewards_episodes.txt'
    np.savetxt(rewardsfile,np.array(all_rewards))

    rewardsfile = 'total_steps_episodes.txt'
    np.savetxt(rewardsfile,np.array(total_epi_counts))

    print("Mean Reward Over 100 Episodes: ", np.mean(np.array(all_rewards)), "+/-", np.std(np.array(all_rewards)))
    print("Mean Total Episodes Counts: ", np.mean(np.array(total_epi_counts)), "+/-", np.std(np.array(total_epi_counts)))

    my_dqn_agent.learn_network.save_model('learning_network.h5')
    my_dqn_agent.target_network.save_model('target_network.h5')


# Running a trained model and recording video

num_episodes             = 1000
my_dqn_agent             = dqn_agent(gamma, steps_per_episode, target_update_freq, mem_dimension, n_minibatches, save_model)
trained_network          = keras.models.load_model('learning_network.h5', custom_objects=None, compile=True)
trained_network_weights  = trained_network.get_weights()

test_rewards, test_run_lengths = my_dqn_agent.run_trained_model(trained_network_weights, num_episodes,  record_video=True)

print("Mean Reward Over 100 Episodes: ", np.mean(np.array(test_rewards)), "+/-", np.std(np.array(test_rewards)) )
print("Mean Total Episodes Counts: ", np.mean(np.array(test_run_lengths)), "+/-", np.std(np.array(test_run_lengths)) )



