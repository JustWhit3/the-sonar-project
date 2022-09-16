#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:51:00 2022
Author: Gianluca Bianco
"""

#################################################
#     Libraries
#################################################

# Generic modules
import doctest
import os

# Data science
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

#################################################
#     save_img
#################################################
def save_img( img_name, save_path, tight = False ):
    """
    Function used to save a specific plot "img_name" into a specific directory "save_path".

    Args:
        img_name (str): The name of the plot to be saved. No file extension needed.
        save_path (str): The path in which the plot should be saved.
        tight (bool, optional): Set "tight" option for plot into True or False. Default to False.
    
    Testing:
        >>> a = [ 1, 2, 3, 4 ]
        >>> _ = plt.plot( a )
        >>> save_img( "save_img_test", "../img/tests" )
        >>> os.path.exists( "../img/tests/save_img_test.png" )
        True
    """
    
    # Create the path if it doesn't exist yet
    if not os.path.exists( save_path ):
        os.makedirs( save_path )

    # Save the plot
    if tight == True:
        plt.savefig( "{}/{}.png".format( save_path, img_name ), bbox_inches = "tight", dpi = 100 )
    elif tight == False:
        plt.savefig( "{}/{}.png".format( save_path, img_name ), dpi = 100 )

#################################################
#     plot_learning_curve
#################################################
def plot_learning_curve( estimator, title, X, y, axes = None, ylim = None, cv = None, n_jobs = None, scoring = None, train_sizes = np.linspace( 0.1, 1.0, 5 ), ):
    """
    Generate 3 plots: the test and training learning curve, the training samples vs fit times curve, the fit times vs score curve. Taken from here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html.

    Args:
        estimator (sklearn): An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        title (str): Title for the chart.
        X (numpy.array): array-like of shape (n_samples, n_features). Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.
        y (numpy.array): array-like of shape (n_samples) or (n_samples, n_features). Target relative to ``X`` for classification or regression;
        axes (numpy.array, optional): array-like of shape (3,). Axes to use for plotting the curves. Defaults to None.
        ylim (numpy.array, optional): tuple of shape (2,). Defines minimum and maximum y-values plotted, e.g. (ymin, ymax). Defaults to None.
        cv (int, optional): cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
        Possible inputs for cv are:. Defaults to None.
        n_jobs (int, optional): nt or None. Number of jobs to run in parallel. Defaults to None.
        scoring (str, optional): a str (see model evaluation documentation) or a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.. Defaults to None.
        train_sizes (numpy.array, optional): array-like of shape (n_ticks,). Relative or absolute numbers of training examples that will be used to generate the learning curve.. Defaults to np.linspace( 0.1, 1.0, 5 ).
    """

    if axes is None:
        _, axes = plt.subplots( 1, 3, figsize = ( 20, 5 ) )

    axes[0].set_title( title )
    if ylim is not None:
        axes[0].set_ylim( *ylim )
    axes[0].set_xlabel( "Training examples" )
    axes[0].set_ylabel( "Score" )

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve( estimator, X, y, scoring = scoring, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes, return_times = True, )
    train_scores_mean = np.mean( train_scores, axis = 1 )
    train_scores_std = np.std( train_scores, axis = 1 )
    test_scores_mean = np.mean( test_scores, axis = 1 )
    test_scores_std = np.std( test_scores, axis = 1 )
    fit_times_mean = np.mean( fit_times, axis = 1 )
    fit_times_std = np.std( fit_times, axis = 1 )

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between( train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r", )
    axes[0].fill_between( train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = "g", )
    axes[0].plot( train_sizes, train_scores_mean, "o-", color = "r", label = "Training score" )
    axes[0].plot( train_sizes, test_scores_mean, "o-", color = "g", label = "Cross-validation score" )
    axes[0].legend( loc = "best" )

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot( train_sizes, fit_times_mean, "o-" )
    axes[1].fill_between( train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha = 0.1, )
    axes[1].set_xlabel( "Training examples" )
    axes[1].set_ylabel( "fit_times" )
    axes[1].set_title( "Scalability of the model" )

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[ fit_time_argsort ]
    test_scores_mean_sorted = test_scores_mean[ fit_time_argsort ]
    test_scores_std_sorted = test_scores_std[ fit_time_argsort ]
    axes[2].grid()
    axes[2].plot( fit_time_sorted, test_scores_mean_sorted, "o-" )
    axes[2].fill_between( fit_time_sorted, test_scores_mean_sorted - test_scores_std_sorted, test_scores_mean_sorted + test_scores_std_sorted, alpha=0.1, )
    axes[2].set_xlabel( "fit_times" )
    axes[2].set_ylabel( "Score" )
    axes[2].set_title( "Performance of the model" )
    
    return plt

#################################################
#     Main
#################################################
if __name__ == "__main__":
    doctest.testmod()