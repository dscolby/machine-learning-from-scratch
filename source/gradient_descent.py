# Author: Darren Colby
# Date: 3/15/2022
# Purpose: To implement minibatch gradient descent from scratch

from autograd import grad
import autograd.numpy as np


def minibatch(g, alpha, max_its, w, x, y, batch_size=32):
    
    """Finds optimal weights by using minibatch gradient descent
       ---------------------------------------------------------
       
       Parameters:
           
            g: a cost function to optimize
            alpha: the learning rate
            max_its: the maximum number of iteratiions to run gradient descent
            w: a numpy array of starting weights
            y: actual values
            
        Returns a tuple of the weight and cost histories"""
    
    # Take a random sample of the data
    indices = np.random.randint(low=x.size, size=batch_size)
    x_sample = x[:, indices]
    y_sample = y[0, indices]
    
    # Weights to be updated
    wk = w
    
    # Calculates the gradient for the above function given any w
    gradient = grad(g)
    
    # Will hold the costs and weights after each iteration
    cost_history = [g(w, x_sample, y_sample)]
    weight_history = [wk]
    
    # Claculates weights by moving down the gradient
    for iter in range(max_its):
        
        # Randomly sample the data
        indices = np.random.randint(low=x.size, size=batch_size)
        x_sample = x[:, indices]
        y_sample = y[0, indices]
        
        wk = wk - alpha*gradient(wk, x_sample, y_sample)
        cost_history.append(g(wk, x_sample, y_sample))
        weight_history.append(wk)
        
    return weight_history, cost_history