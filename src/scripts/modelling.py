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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

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
#     MAE_vs_fold
#################################################
def MAE_vs_fold( kfold, X, Y ):
    
    mae_train = []
    mae_test = []
    for train_index, test_index in kfold.split(X):

       X_train, X_test = X.iloc[train_index], X.iloc[test_index]
       y_train, y_test = y[train_index], y[test_index]
       model = KNeighborsClassifier(n_neighbors=2)
       model.fit(X_train, y_train)
       y_train_pred = model.predict(X_train)
       y_test_pred = model.predict(X_test)
       mae_train.append(mean_absolute_error(y_train, y_train_pred))
       mae_test.append(mean_absolute_error(y_test, y_test_pred))

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
    MAE_vs_fold( kfold, X, Y )
    
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