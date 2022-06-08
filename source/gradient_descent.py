# Author: Darren Colby
# Date: 3/20/2022
# Purpose: To implement gradient descent from scratch

from autograd import grad
import autograd.numpy as np


def gradient_descent(g, learning_rate, max_its, w, x, y, regularization=None, 
                     lmbda=0.005, alpha=0.5, flavor='batch', batch_size=32, 
                     random_state=None):
    
    """Finds optimal weights by using gradient descent
       ---------------------------------------------------------
       
       Parameters:
           
            g: a cost function to optimize
            alpha: the learning rate
            max_its: the maximum number of iteratiions to run gradient descent
            w: a numpy array of starting weights
            x: a numpy array of features
            y: actual values
            regularization: type of penalty to use
            lmbda: coefficient for the penalty
            alpha: weight of the L1 penalty if using ElasticNet
            flavor: either batch or sgd
            batch_size: the batch size if using stochastic gradient descent
            random_state: a number to set for reproducibility
            
        Returns a tuple of the weight and cost histories"""
        
    # Safety check
    if flavor not in ['batch', 'sgd']:
        raise ValueError('flavor must be one of batch or sgd')
        
    # Set a seed
    if random_state is not None:
        np.random.RandomState(seed=random_state)
        
    # Perform gradient descent on the whole dataset
    if flavor == 'batch':
        x_sample = x
        y_sample = y
        
    # SGD
    else:
        
        # Take a random sample of the data
        indices = np.random.randint(low=x.size, size=batch_size)
        x_sample = x[:, indices]
        y_sample = y[0, indices]
    
    # Weights to be updated
    wk = w
    
    # Calculates the gradient for the above function given any w
    gradient = grad(g)
    
    # Will hold the costs and weights after each iteration
    cost_history = [g(w, x_sample, y_sample, regularization, lmbda, alpha)]
    weight_history = [wk]
    
    # Calculates weights by moving down the gradient
    for iter in range(max_its):
        
        if flavor == 'sgd':
            
            # Randomly sample the data
            indices = np.random.randint(low=x.size, size=batch_size)
            x_sample = x[:, indices]
            y_sample = y[0, indices]
        
        wk = wk - learning_rate * gradient(wk, x_sample, y_sample)
        cost_history.append(g(wk, x_sample, y_sample))
        weight_history.append(wk)
        
    return weight_history, cost_history