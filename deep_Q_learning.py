import numpy as np
import random as rand
import gym
import matplotlib.pyplot as plt
import replay_fn as rpf
import neural_network as qnet
import tensorflow as tf

env = gym.make('LunarLander-v2')

#alpha = 0.5
rar = 0.2 
radr = 0.995       
gamma = 0.99
mem_dimension = 100000
num_actions = 4
episodes = 200
steps_per_episode = 1000
n_minibatches = 100
target_update_freq = 200
save_model = 200

# Initialize replay memory D
D = rpf.reply_fn(mem_size=mem_dimension,batch_size=n_minibatches)

# Initialize learning Q neural network using random weights
learn_network = qnet.neural_network(hidden1=64, hidden2=128, hidden3=64, n_classes=4, batch_size=100,nvariables=8)

# Initialize target Q neural using the weights of the learning network
target_network = qnet.neural_network(hidden1=64, hidden2=128, hidden3=64, n_classes=4, batch_size=100,nvariables=8)
target_network.copy_weights(learn_network) 

def run_one_episode(sess,rar,radr,gamma):
    
    episode_reward = 0
    state = env.reset()

    count = 0
    while count <  steps_per_episode:

        random_or_optimal = rand.random()
        if random_or_optimal < rar:
            action = rand.randint(0, num_actions-1)
        else:
            # Generate action using learning network
            action = np.argmax(learn_network.test_neural_network(sess,np.reshape(state, (1,-1))) )

        # Excute action in emulator and observe reward and next state
        s_prime,R, isTerminal, info =  env.step(action)
        episode_reward +=  R
 
        # Store rewards in memory D
        D.add_to_memory(state,R,action,s_prime,isTerminal)
        state = s_prime

	if isTerminal:
	    break

        # Sample mini-batches from memory if count is greater than equilibration steps
        memory_samples = D.replay_memory(n_minibatches)

        if memory_samples is not None:
            
            b_states, b_actions, b_Rs, b_s_primes, b_isTerminals = memory_samples

            #  Do greedy policy in this batches
            targets = learn_network.test_neural_network(sess,np.array(b_states))
	    
            q_prime = target_network.test_neural_network(sess,np.array(b_s_primes))
            q_prime = np.max(q_prime,axis=1)
	    q_prime = np.array(b_Rs) + gamma*q_prime*np.array(b_isTerminals)

	    targets[np.arange(len(b_actions)), b_actions] = q_prime            
	    # Do one step of gradient descent
            learn_network.train_neural_network(sess,b_states, targets)
        if steps_per_episode % target_update_freq == 0:
            # Copy weights from learning network to target network
	    target_network.copy_weights(learn_network)
	count += 1
    
    return episode_reward

episode_count = 0.0
all_rewards = []
i=0

while True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode_rewards = run_one_episode(sess,rar,radr,gamma)
        print "Episode ", str(i+1)+":", episode_rewards
        episode_count += 1.0
        all_rewards.append(episode_rewards)
        if rar > 0.01:
            rar = rar*radr
        if episode_count % save_model == 0:
            target_prefix = 'target_nn_%i' % i
            learning_prefix = 'learning_nn_%i' % i
            learn_network.save_model(sess,learning_prefix)
            target_network.save_model(sess,target_prefix)
        if episode_rewards >= 200:
            target_prefix2 = 'conv_target_nn_%i' % i
            learning_prefix2 = 'conv_learning_nn_%i' % i
            learn_network.save_model(sess,learning_prefix2)
            target_network.save_model(sess,target_prefix2)
            break 
        i += 1

#Plot learning
plt.plot(all_rewards, marker = '.', ms=14) 
plt.plot(all_rewards, ls = '-', ms=8)
#plt.ylim([0.15,0.8])
#plt.xlim([-0.1,1.2])
#plt.xlabel(r'$\mathrm{\alpha}$', family='sans-serif',size='14',stretch='ultra-condensed',color='k')
plt.xlabel('Training Epoch')
plt.ylabel('Rewards')
plt.title('Deep Q-Learning: Lunar Lander', family='sans-serif',size='18',stretch='ultra-condensed',color='r')
plt.savefig('rewards.png')
plt.savefig('rewards.pdf')

#Save reward sequence to file
rewardsfile = 'rewards_'+learning_prefix2+'.txt'
np.savetxt(rewardsfile,np.array(all_rewards))

#Restore model from datafiles
#with tf.Session() as sess:
#    saver = tf.train.import_meta_graph('filename.meta')
#    saver.restore(sess, "checkpoint_file")

