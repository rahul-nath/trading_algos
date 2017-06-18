"""
QLearner implementation (c) 2017 Rahul Nath, Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.0, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        # seed the RNG
        #np.random.seed(num_states)

        self.q = np.random.uniform(low=-1.0, size=(num_states, num_actions))
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.alpha = alpha
        self.gamma = gamma
        self.state_reward = []
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.unravel_index(self.q[s, :].argmax(), self.q[s, :].shape)[0]
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self, s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        # update the Q table (using the max action) 
        first_action = np.unravel_index(self.q[s_prime, :].argmax(), self.q[s_prime, :].shape)[0]
        self.q[self.s, self.a] = self.q[self.s, self.a] + self.alpha*(r + self.gamma * self.q[s_prime, first_action] - self.q[self.s, self.a])
        self.state_reward.append((self.s, self.a, s_prime, r))

        # generate random numbers instead of the random number each time
        for _ in xrange(self.dyna):
            # for list of experienced action, state, new_state, reward pairs, choose one
            index = np.random.choice(len(self.state_reward))
            new_s, new_a, new_state, new_reward = self.state_reward[index]
            max_action = np.unravel_index(self.q[new_state, :].argmax(), self.q[new_state, :].shape)[0]
            self.q[new_s, new_a] += self.alpha*(new_reward + self.gamma * self.q[new_state, max_action] - self.q[new_s, new_a])

        new_rar = np.random.uniform()
        if  new_rar< self.rar:
            self.a = rand.randint(0, self.num_actions-1)
        else:
            self.a = first_action
        self.rar *= self.radr

        self.s = s_prime        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        return self.a

    def author(self):
        return "rnath9"

if __name__=="__main__":
    print "nothing"

