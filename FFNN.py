"""
General OpenAI gym player
Simple feed forward neural net that solves OpenAI gym environments 
(https://gym.openai.com) via Q-learning 
Download the code, assign to game_name the name of environment you wish to 
run, and let the script learn how to solve it.  
Note the code only works for environments with discrete action space and 
continuous observation space.
https://github.com/FlankMe/general-gym-player
@author: Riccardo Rossi

Edited 2017, Rahul Nath
"""

# Choice of the game and definition of the goal
game_name = 'CartPole-v0'
MAX_EPISODES = 1000
CONSECUTIVE_EPISODES = 100   # Number of trials' rewards to average for solving
IS_RECORDING = False 

# Fine-tuning the EPSILON_DECAY parameters will lead to better results for 
# some environments and worse for others. As this code is a go at a 
# general player, it is neater to treat it as a global constant 
EPSILON_DECAY = 0.99

# Import basic libraries
import numpy as np


"""
Plain Feed Forward Neural Network
The chosen activation function is the Leaky ReLU function
"""
class FeedForwardNeuralNetwork:
    
    def __init__(self, layers):

        # NN variables
        self._generateNetwork(np.array(layers))


    def _generateNetwork(self, layers):
        """
        The network is implemented in Numpy
        Change this method if you wish to use a different library
        """
        
        self._ALPHA = 1e-2
        # Activation function used is the Leaky ReLU function
        self._activation = lambda x : x * (0.01*(x<0) + (x>=0))
        self._derive = lambda x : 0.01*(x<0) + (x>=0)
        
        # Initialization parameters
        INITIALIZATION_WEIGHTS = 0.1
        INITIALIZATION_BIAS = -0.001

        # Create the graph's architecture
        self._weights = []
        self._bias = []

        for i in range(layers.size - 1):
            weight = np.random.uniform(-INITIALIZATION_WEIGHTS, 
                                       INITIALIZATION_WEIGHTS,
                                       size=(layers[i], layers[i+1]))
            bias = INITIALIZATION_BIAS * np.ones((layers[i+1]))
            self._weights.append(weight)
            self._bias.append(bias)
            
    def _feedFwd(self, X):

        self._activation_layers = [np.atleast_2d(X)]
        
        for i in range(len(self._weights) - 1):
            self._activation_layers.append(self._activation(
                np.dot(self._activation_layers[-1], self._weights[i]) + self._bias[i]))
                
        # Last layer does not require the activation function
        self._activation_layers.append(
            np.dot(self._activation_layers[-1], self._weights[-1]) + self._bias[-1])
        
        return(self._activation_layers[-1])          

    def _backProp(self, X, a, y):    
        # Calculate the delta vectors
        self._delta_layers = [a * (np.atleast_2d(y).T - self._feedFwd(X))]
        
        for i in range(len(self._activation_layers) - 2, 0, -1):
            self._delta_layers.append(np.dot(self._delta_layers[-1], self._weights[i].T) * self._derive(self._activation_layers[i]))  
        self._delta_layers.reverse()
        
        # Reduce the learning rate if the error grows eccessively
        if np.array([np.abs(delta).sum() for delta in self._delta_layers]).sum() > 1. / self._ALPHA:
            self._ALPHA /= 2.
            
        # Update the weights and bias vectors
        for i in range(len(self._weights)):
            self._weights[i] += self._ALPHA * np.dot(self._activation_layers[i].T, self._delta_layers[i])
            self._bias[i] += self._ALPHA * self._delta_layers[i].sum(axis=0)
                                       
                                         
    def predict(self, state):    
        return(self._feedFwd(state))
       
    def fit(self, valueStates, actions, valueTarget, weights=None):                      
        self._backProp(valueStates, actions, valueTarget)
