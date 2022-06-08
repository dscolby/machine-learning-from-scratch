# Author: Darren Colby
# Date: 3/21/2022
# Purpose: To provide functions for L! and L2 regularization

import autograd.numpy as np


def L1(weights):
    """Calculates the L1 norm of a vector
       ----------------------------------
       
       Parameters:
           
           weights: a numpy array of weights
           
       Returns the L1 norm of the weight vector"""
    return np.linalg.norm(weights, ord=1)


def L2(weights):
    """Calculates the square of the L2 norm of a vector
       ------------------------------------------------
       
       Parameters:
           
           weights: a numpy array of weights
           
       Returns the square of the L2 norm of the weight vector"""
    return np.linalg.norm(weights, ord=2)**2


def elasticnet(weights, alpha=0.5):
    """Calculates an Elastic Net penalty
       ----------------------------------
       
       Parameters:
           
           weights: a numpy array of weights
           alpha: the weight of the L1 penalty
           
       Returns the Elastic Net penalty of the weight vector"""
       
    return (alpha * L1(weights)) + ((1 - alpha) * L2(weights))

