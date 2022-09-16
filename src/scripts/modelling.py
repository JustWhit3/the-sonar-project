#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Sep 15 15:48:00 2022
Author: Gianluca Bianco
"""

#################################################
#     Libraries
#################################################

# Generic modules
from termcolor import colored as cl
import sys
import argparse as ap

# Data science
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Personal modules
sys.path.append( ".." )
from utils.generic import save_img

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

#################################################
#     box_plot
#################################################
def box_plot( names, scores, label ):

    fig = plt.figure()
    fig.suptitle( "Algorithms comparison ({})".format( label ) )
    ax = fig.add_subplot( 111 )
    plt.boxplot( scores )
    ax.set_xticklabels( names )
    save_img( label.replace( " ", "_" ), "{}/box_plots".format( model_path ) )

#################################################
#     splitting_dataset
#################################################
def modelling( model, str_name, X, Y ):
    
    # Using k-Fold cross-validation for train/test splitting
    print( "Model:", cl( str_name, "green" ) )
    num_folds = int( args.n_of_folds )
    kfold = KFold( n_splits = num_folds, random_state = 7, shuffle = True )
    
    # Plotting learning curve to check for possible overfitting
    plot_learning_curve( model, str_name, X, Y )
    save_img( str_name.replace( " ", "_" ), "{}/learning_curves".format( model_path ) )
    
    # Accuracy
    result_acc = cross_val_score( model, X, Y, cv = kfold, scoring = "accuracy" )
    print( "- Accuracy: ", end = "" )
    print( cl( "%.3f%% +/- %.3f%%" % ( result_acc.mean()*100.0, result_acc.std()*100.0 ), "yellow" ) )

    # Negative log-loss
    result_nll = cross_val_score( model, X, Y, cv = kfold, scoring = "neg_log_loss" )
    print( "- Negative log-loss: ", end = "" )
    print( cl( "%.3f +/- %.3f" % ( result_nll.mean(), result_nll.std() ), "yellow" ) )

    # Area under the roc curve
    result_auc = cross_val_score( model, X, Y, cv = kfold, scoring = "roc_auc" )
    print( "- Area under the ROC curve: ", end = "" )
    print( cl( "%.3f +/- %.3f" % ( result_auc.mean(), result_auc.std() ), "yellow" ) )

    # Confusion matrix
    print( "- Saving confusion matrix" )
    predicted = cross_val_predict( model, X, Y, cv = kfold )
    matrix = confusion_matrix( Y, predicted )
    df_cm = pd.DataFrame( matrix )
    plt.figure( figsize = ( 10,7 ) )
    sn.heatmap( df_cm, annot = True, cmap = "YlOrRd", fmt = "d" )
    plt.title( str_name )
    save_img( str_name.replace( " ", "_" ), "{}/confusion_matrix".format( model_path ) )
    
    return result_acc, result_nll, result_auc

#################################################
#     Main
#################################################
def main():
    
    # Variables
    global model_path
    model_path = "../../img/modelling"
    results_acc = []
    results_nll = []
    results_auc = []
    
    # Loading the dataset
    data = pd.read_csv( args.data )
    dim = data.shape[1] - 1
    array = data.values
    X = array[ :, 0:dim ]
    Y = array[ :, dim ]

    # Try different models
    models = []
    models.append( LogisticRegression( solver = "lbfgs", max_iter = 500 ) )
    models.append( LinearDiscriminantAnalysis() )
    models.append( KNeighborsClassifier() )
    models.append( DecisionTreeClassifier() )
    models.append( GaussianNB() )
    models.append( SVC( gamma = "scale", probability = True ) )
    for index, model_name in enumerate( models ):
        if index != 0:
            print()
        r_ = modelling( model_name, type( model_name ).__name__, X, Y )
        results_acc.append( r_[0] )
        results_nll.append( r_[1] )
        results_auc.append( r_[2] )
    
    # Doing box plots
    names = [ "LR", "LDA", "KNN", "CART", "NB", "SVM" ]
    box_plot( names, results_acc, "accuracy" )
    box_plot( names, results_nll, "negative log-loss" )
    box_plot( names, results_auc, "area under the ROC curve" )

if __name__ == "__main__":
    
    # Argument parser settings
    parser = ap.ArgumentParser( description = "Argument parser for data preparation." ) 
    parser.add_argument( "--data", default = "../../data/processed_data.csv", help = "Input dataset." )
    parser.add_argument( "--n_of_folds", default = 5, help = "Number of folds used in the k-Fold algorithm. Common values: 3,5 or 10." )
    args = parser.parse_args()

    # Running the program
    main()   