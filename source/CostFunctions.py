# Author: Darren Colby
# Date: 5/2/2022
# Purpose: TO provide cost functions for machine learning models

import autograd.numpy as np
from regularizaers import L1, L2, elasticnet


def MSE(w, x, y, regularization=None, lmbda=0.005, alpha=0.5):
    """Calculates the mean squared error between predicted and actual values
            
            Parameters:
                w: a NumPy array of weights
                x: a NumPy array of features
                y: a NumPy array of actual values
                regularization: type of penalty to use
                lmbda: coefficient for the penalty
                alpha: weight of the L1 penalty if using ElasticNet
                
            Returns:
                the mean squared error
    """

    mse = (np.sum((np.dot(x.T, w[1:]) + w[0]) - y)**2) / y.size
    
    # No regularization
    if regularization is None:
        return mse
    
    # L2 regularization
    if regularization == 'ridge':
        return mse + (lmbda * L2(w))
    
    # L1 regularization
    if regularization == "lasso":
        return mse + (lmbda * L1(w))
    
    # ElasticNet regularization
    if regularization == 'elasticnet':
        return mse + (lmbda * elasticnet(w, alpha))


def MAE(w, x, y, regularization=None, lmbda=0.005, alpha=0.5):
    """Calculates the mean absolute error between predicted and actual values
            
            Parameters:
                w: a NumPy array of weights
                x: a NumPy array of features
                y: a NumPy array of actual values
                regularization: type of penalty to use
                lmbda: coefficient for the penalty
                alpha: weight of the L1 penalty if using ElasticNet
                
            Returns:
                the mean absolute error
    """
    
    mae =  (np.sum((np.abs(x.T, w[1:]) + w[0]) - y)) / y.size
    
    # No regularization
    if regularization is None:
        return mae
    
    # L2 regularization
    if regularization == 'ridge':
        return mae + (lmbda * L2(w))
    
    # L1 regularization
    if regularization == "lasso":
        return mae + (lmbda * L1(w))
    
    # ElasticNet regularization
    if regularization == 'elasticnet':
        return mae + (lmbda * elasticnet(w, alpha))
    