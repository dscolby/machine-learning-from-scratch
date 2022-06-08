# Author: Darren Colby
# Date: 5/7/2022
# Purpose: To estimate linear regressions via gradient descent

import autograd.numpy as np
from gradient_descent import gradient_descent
from CostFunctions import MSE, MAE

class LinearRegression:
    """
    
    
    Estimates linear regression using gradient descent. Can incorporate L1, 
       L2, and elasticnet regularization and use batch or stochastic gradient 
       descent.
       
       ...
       
       Attributes
       ----------
           FIT : boolean
               whether a linear regression has already been fit
               
           weight_history : ndarray
               weights from each iteration of gradient descent
           
           cost_history : ndarray
               cost from each iteration of gradient descent
               
            weights : ndarray
                the fitted weights for teh model
                
            cost : float64
                the cost of the fitted model
               
       Methods
       -------
           __model(w, x):
               calculates the dot product of w and x plus a bias term
               
          fit(x_train, y_train, how='batch', batch_size=32):
              fits a linear model to training data"""
    
    def __init__(self):
        self.FIT = False
        self.weight_history = None
        self.cost_history = None
        self.weights = None
        self.cost = None
    
    def fit(self, x_train, y_train, cost=MSE, how='batch', learning_rate=0.01, 
            batch_size=32, max_its=300, regularization=None, lmbda=0.005, 
            alpha=0.5, seed=None):
        """
        

        Parameters
        ----------
        x_train : ndarray
            Training features.
        y_train : ndarray
            Values to predict.
        cost : callable, optional
            MSE for means squared error or MAE for mean absolute error. 
            The default is MSE.
        how : str, optional
            Type of gradient descent to perform. Options are 'batch' for batch
            gradient descent and 'sgd' for stochastic gradient descent.
            The default is 'batch'.
        learning_rate : float, optional
            the rate of gradient descent. The default is 0.01.
        batch_size : int, optional
            The batch size if performing stochastic gradient descent. 
            The default is 32.
        max_its : int, optional
            The number of iterations to perfomr gradient descent. 
            The default is 300.
        regularization : str, optional
            Choose from 'ridge', 'lasso', or 'elasticnet'. The default is None.
        lmbda : float, optional
            The shrinkage parameter if using regularization. 
            The default is 0.005.
        alpha : float, optional
            weight of the L1 penalty if using ElasticNet regularization. 
            The default is 0.5.
        seed : int, optional
            A seed for reproducibility. The default is None.

        Raises
        ------
        ValueError
            Ensures the cost function is one of MSE or MAE; how to find the 
            weights is one of 'batch' or 'sgd'; and if using regularization, 
            the type is one of 'ridge', 'lasso', or 'elasticnet'.

        Returns
        -------
        None.

        """
        
        # Make sure there is a valid cost function
        if cost not in [MSE, MAE]:
            raise ValueError ("The cost must be one of MSE or MAE")
            
        # Make sure there is a valid type of optimization
        if how not in ['batch', 'sgd']:
            raise ValueError("how must be either batch or sgd")
            
        # make sure the type of regularization is correct
        if regularization not in [None, 'ridge', 'lasso', 'elasticnet']:
            raise ValueError("regularization must be ridge, lasso, or elasticnet")
            
        # Set a seed
        if seed is not None: 
            np.random.RandomState(seed=seed)
            
        # Random starting weights
        starting_weights = np.repeat(np.random.uniform(-0.1, 0.1), 
                                     x_train.shape[1])
        
        # Run gradient descent
        self.weight_history, self.cost_history = gradient_descent(cost,
                                                                  learning_rate, 
                                                                  max_its,
                                                                  starting_weights,
                                                                  x_train, 
                                                                  y_train,
                                                                  regularization,
                                                                  lmbda, alpha,
                                                                  how,
                                                                  batch_size, 
                                                                  random_state=seed)
        
        # Update the weights
        self.weights = self.weight_history[-1]
        
        # Update the cost
        self.cost = self.cost_history[-1]
        
        # enable prediction
        self.FIT = True
        
    def predict(self, x_test):
        """
        

        Parameters
        ----------
        x_test : ndarray
            Test data to make predictions for.

        Returns
        -------
        Predicted values.

        """
        
        return np.dot(x_test.T, self.weights[1:]) + self.weights[0]