import numpy as np
import random as rand

class reply_fn(object):

    def __init__(self, mem_size=20000,batch_size=16):
        self.mem_size   = mem_size
        self.batch_size = batch_size
        self.states     = []
        self.s_primes   = []
        self.rewards    = []
        self.terminal   = []
        self.actions    = []

    def add_to_memory(self,state,R,action,s_prime,isTerminal):

        curr_size = len(self.states)
        if curr_size > self.mem_size:
	    #remove_idx = np.random.randint(0,len(self.states),size=1)
            remove_idx = 0
            self.states.pop(remove_idx)
	    self.s_primes.pop(remove_idx)
	    self.rewards.pop(remove_idx)
            self.terminal.pop(remove_idx)
            self.actions.pop(remove_idx)

        self.states.append(state)
        self.rewards.append(R)
        self.actions.append(action)
        self.s_primes.append(s_prime)
        self.terminal.append(isTerminal)

    def replay_memory(self,nsamples):
        if len(self.states) > self.batch_size:
            memory_indexes = np.random.choice(len(self.states), size=nsamples, replace=True)
            memory = [ np.asarray(self.states)[memory_indexes],np.asarray(self.actions)[memory_indexes], np.asarray(self.rewards)[memory_indexes], \
                     np.asarray(self.s_primes)[memory_indexes], np.asarray(self.terminal)[memory_indexes] ]
            return memory
        return None
