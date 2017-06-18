from numpy import random
import numpy as np

"""
Implementation of Adele Cutler's Random Tree Learner,
(https://www.researchgate.net/publication/268424569_PERT-perfect_random_tree_ensembles)

Something wrong with the query method -- learner should overfit perfectly with in-sample
data, but currently has only 92% - 94% accuracy.
"""

class RTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = []
        self.leaf_size = leaf_size
    
    def isSame(self, a):
        b = a == a[0]
        return np.all(b)

    def addEvidence(self, data_x, data_y):
        # make an ndarray combining the x and y data
        combined_data = np.append(data_x, np.array([data_y]).T, axis=1)
        # then refer back to the pseudocode

        def build_tree(data):
            if not data.shape:
                return np.array([[-1, -1, None, None]])
            if data.shape[0] <= self.leaf_size:
                # condense the y values into one
                avg = data[:, -1].mean()
                if np.isnan(avg):
                    avg = 0
                return np.array([[-1, avg, None, None]])
            if self.isSame(data[:,-1]):
                return np.array([[-1, data[0:, -1], None, None]])                
            i = random.choice(data.shape[1] - 1)
            rand_row1 = random.choice(data.shape[0]-1, replace=False)
            rand_row2 = random.choice(data.shape[0]-1, replace=False)
            try_count = 0
            while rand_row1 == rand_row2:
                if try_count == 3:
                    # choose one as the split value
                    #break
                    return np.array([[-1, data[:, -1].mean(), None, None]])
                rand_row2 = random.choice(data.shape[0]-1, replace=False)
                try_count += 1

            split_val = (data[rand_row1, i] + data[rand_row2, i])/2
            left = build_tree(data[data[:, i] <= split_val, :]) 
            right = build_tree(data[data[:, i] > split_val, :]) 
            root = np.array([[i, split_val, 1, left.shape[0]+1]])
        
            return np.vstack((root, left, right))
        self.tree = build_tree(combined_data)
        #print self.tree

    def author(self):
        return "rnath9"

    def query(self, test_data):

        result = []
        for row in test_data:
            index = 0
            while True:
                feature_i = self.tree[index][0]
                split = self.tree[index][1]
                if feature_i == -1:
                    result.append(split)
                    break
                elif row[feature_i] <= split:
                    index += self.tree[index][2]
                else: 
                    index += self.tree[index][3]
        return result
