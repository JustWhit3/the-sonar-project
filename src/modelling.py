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
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix
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
from utils import save_img, plot_learning_curve

#################################################
#     box_plot
#################################################
def box_plot( names, scores, label ):
    """
    Function used to produce box plot of "scores" with "names" features.

    Args:
        names (list): list of names to be used as feature labels.
        scores (list): list of scores to be used as bar plots.
        label (str): metric type used for box plots.
    """

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
    """
    Function used to test models with different metrics and plotting learning curves.

    Args:
        model (sklearn): the machine learning model.
        str_name (str): the string name of the machine learning model.
        X (np.array): array of input data.
        Y (np.array): array of labels used for classification.

    Returns:
        np.array: output results of each metric.
    """
    
    # Using k-Fold cross-validation for train/test splitting
    print( "Model:", cl( str_name, "green" ) )
    num_folds = int( args.n_of_folds )
    kfold = KFold( n_splits = num_folds, random_state = 7, shuffle = True )

    # Plotting learning curves for accuracy
    plot_learning_curve( model, "{} (accuracy)".format( str_name ), X, Y, cv = kfold, scoring = "accuracy" )
    save_img( str_name.replace( " ", "_" ), "{}/learning_curves/accuracy".format( model_path ) )
    
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
#     hyperparametrization
#################################################
def hyperparametrization( model, X, Y ):
    """
    Function used to test different parameters combinations (and choosing the best one) for each model.

    Args:
        model (sklearn): the machine learning model.
        X (np.array): input array of data.
        Y (np.array): input array of binary data used for classification.

    Returns:
        sklearn: return model with best hyperparameter combination.
    """

    # Setting parameters for different models
    if type( model ).__name__ == "KNeighborsClassifier":
        param_grid = { 
            "n_neighbors": np.arange( 1, 30 ),
            "algorithm": [ "auto", "ball_tree", "kd_tree", "brute" ],
            "metric": [ "euclidean", "manhattan", "chebyshev", "minkowski" ]
            }
    elif type( model ).__name__ == "LinearDiscriminantAnalysis":
        param_grid = { 
            "solver": [ "svd", "lsqr", "eigen" ]
            }
    elif type( model ).__name__ == "DecisionTreeClassifier":
        param_grid = { 
            "criterion": [ "gini", "entropy", "log_loss" ],
            "splitter": [ "best", "random" ],
            }
    elif type( model ).__name__ == "SVC":
        param_grid = { 
            "degree": np.arange( 1, 10 ),
            "gamma": [ "scale", "auto" ]
            }
    elif type( model ).__name__ == "GaussianNB":
        param_grid = {}
    elif type( model ).__name__ == "LogisticRegression":
        param_grid = {}
       
    # Applying grid search 
    num_folds = int( args.n_of_folds )
    kfold = KFold( n_splits = num_folds, random_state = 7, shuffle = True )
    grid = GridSearchCV( model, param_grid = param_grid, cv = kfold )
    grid.fit(X, Y)
    best_estimator = grid.best_estimator_
    
    return best_estimator

#################################################
#     Main
#################################################
def main():
    
    # Variables
    global model_path
    model_path = "../img/modelling"
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
    models.append( LogisticRegression( max_iter = 500, penalty = "l1", solver = "liblinear" ) )
    models.append( LinearDiscriminantAnalysis() )
    models.append( KNeighborsClassifier( metric = "manhattan", n_neighbors = 1 ) )
    models.append( DecisionTreeClassifier() )
    models.append( GaussianNB() )
    models.append( SVC( gamma = "scale", probability = True, degree = 1 ) )
    for index, model_name in enumerate( models ):
        if index != 0:
            print()
        r_ = modelling( model_name, type( model_name ).__name__, X, Y )
        results_acc.append( r_[0] )
        results_nll.append( r_[1] )
        results_auc.append( r_[2] )
        if args.hyperparametrization == "on":
            print( "Hyperparametrization:", hyperparametrization( model_name, X, Y ) )
    
    # Doing box plots
    names = [ "LR", "LDA", "KNN", "CART", "NB", "SVM" ]
    box_plot( names, results_acc, "accuracy" )
    box_plot( names, results_nll, "negative log-loss" )
    box_plot( names, results_auc, "area under the ROC curve" )

if __name__ == "__main__":
    
    # Argument parser settings
    parser = ap.ArgumentParser( description = "Argument parser for data preparation." ) 
    parser.add_argument( "--data", default = "../data/processed_data.csv", help = "Input dataset." )
    parser.add_argument( "--n_of_folds", default = 5, help = "Number of folds used in the k-Fold algorithm. Common values: 3,5 or 10." )
    parser.add_argument( "--hyperparametrization", default = 5, help = "Display hyperparametrization studies (on/off)." )
    args = parser.parse_args()

    # Running the program
    main()   