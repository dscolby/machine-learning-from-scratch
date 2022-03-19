# Author: Darren Colby
# Date: 3/18/2022
# Purpose: To implement a Naive Bayes classifier

import numpy as np


class NaiveBayes:
    """Enables fitting a Naive Bayes model to training data and using the
       fitted model to predict class labels. This implementation will work for
       binary and multiclass classification and can use either a Gaussian or 
       multinomial probability distribution of the features.
    """
    
    def __init__(self):
        self.priors_dict = {} # Will store priors for each class
        self.labels = None # Will hold class labels
        self.FIT = False # False if fit has not been called
        
    def fit(self, y_train):
        """Calculates the prior for each class
           -----------------------------------
       
           Parameters:
               
               y_train: 1D array of classes to predict from training data
        
           Returns dictionary where keys are classes and values are priors"""
        
        # The unique classes in the dataset
        self.labels = np.unique(y_train)
    
        # Each class is a key and its prior is the value
        priors_dict = {label:np.log(y_train[y_train == label].size / 
                                    y_train.size) for label in self.labels}
            
        self.priors_dict = priors_dict
            
        # Enables predict method to be called
        self.FIT = True
    
        return priors_dict
    
    def __gaussian_log_likelihood(self, row, subsetted_data, eps):
        
        """Calculates and sums the log of posteriors for each feature in a row 
           of data in a numpy array from the Gausssian PDF
           -------------------------------------------------------------------
       
          Parameters:
              
              row: the row of data to calculate the gaussian log-likelihood for
              subsetted_data: a 2D numpy array of data subsetted to a class of 
                              interest
              eps: a constant to add to probabilities before taking their log 
                   to avoid dividing by zero
           
          Returns the Gaussian log-likelihood for the given row of data"""
    
        # The sum of log probabilities for each feature
        total_log_prob = 0
    
        # Loop through each feature of the data point
        for num in row:
        
            # Get the index in the array. This corresponds to the column to get 
            # the mean and standard deviation of
            idx = np.where(row == num)
        
            # Parameters of the normal distribution in the column corresponding 
            # to the index 
            mean = np.mean(subsetted_data[:, idx])
            std_dev = np.std(subsetted_data[:, idx])
        
            # Add the log of the probability from the Gaussian PDF
            term_1 = ((1.0 / np.sqrt(2.0 * np.pi * std_dev**2)))
            term_2 = np.exp(-((num - mean)**2) / (2 * (std_dev**2)))          
            total_log_prob += np.log((term_1 * term_2) + eps)
    
        return total_log_prob
        
    def __multinomial_log_likelihood(self, row, subsetted_data, smooth, eps):
        """Calculates and sums the log of posteriors for each feature in a row 
           of data in a numpy array from the Gausssian PDF
           -------------------------------------------------------------------
       
          Parameters:
              
              row: row of data to calculate the multinomial log-likelihood for
              subsetted_data: a 2D numpy array of data subsetted to a class of 
                              interest
              smooth: a smoothing parameter for Laplace smoothing
              eps: a constant to add to probabilities before taking their log 
                   to avoid dividing by zero
           
         Returns the multinomical log-likelihood for the given row of data"""
    
        # The sum of the log probabilities for each feature
        total_log_prob = 0
    
        # Loop through each feature in each data point
        for num in row:
       
            # Get the index of the array, which corresponds to the column to 
            # use for the multinomial PDF
            idx = np.where(row == num)
        
            # Find the number of observations with the same value and all 
            # values in the training data subsetted by class
            same_obs = np.where(subsetted_data[:, idx] == num)[0].size + smooth
            number_obs = subsetted_data[:, idx].size + subsetted_data.shape[1]
        
            # Update the total log likelihood
            total_log_prob += np.log((same_obs / number_obs) + eps)
        
        return total_log_prob
        
    def predict(self, x_train, y_train, x_test, pdf="gaussian", smooth=1.0, 
                eps=1e-9):
        """Predicts a Naive Bayes classifier
        ---------------------------------
       
           Parameters:
               
               x_train: input features for the training data set
               y_train: class labels for the training data set
               x_test: input features from the test data to predict the labels
               pdf: the probability density fuction to use to calculate the 
                    likelihood; options are guassian and multinomial
               smooth: a smoothing parameter for Laplace smoothing; if pdf is 
                       Gaussian this prameter is ignored
               eps: a constant to add to probabilities before taking their log 
                    to avoid dividing by zero
           
           Returns a tuple of predicted class labels and the log posteriors for 
                   each class and data point"""
    
        # Safety check
        if pdf not in ["multinomial", "gaussian"]:
            raise ValueError("""This Naive Bayes implementation currently only 
                                supports multinomial and Gaussian distributions
                                """)
        
        # Make fit has been called
        if not self.FIT:
            raise RuntimeError("""The fit method must be called before calling 
                                  predict""")
    
        # Get the priors
        priors = self.fit(y_train)
    
        # Stores posteriors
        posteriors_list = []
    
        # Stores predictions
        predictions = []
    
        # Loop through each unique class
        for label in self.labels:
        
            # Data subsetted to each label
            subset = x_train[y_train == label]
        
            # Applies the log likelihood to every element in every row of the 
            # test data and calculates its posterior
            if pdf == "gaussian":
                posteriors = (priors[label] + 
                              np.apply_along_axis(
                                  self.__gaussian_log_likelihood, 1, 
                                  x_test, subset, eps)).tolist()
        
            if pdf == "multinomial":
                posteriors = (priors[label] + 
                              np.apply_along_axis(
                                  self.__multinomial_log_likelihood, 
                                                  1, x_test, subset, 
                                                  smooth, eps)).tolist()
            
            posteriors_list.append(posteriors)
        
        # Loop through posteriors to find the class with the highest posterior
        for idx in np.apply_along_axis(np.argmax, 1, 
                                       np.array(posteriors_list).T):
            
            # Add the corresponding label
            predictions.append(int(self.labels[idx]))
        
        return np.array(predictions), np.array(posteriors_list).T
        
