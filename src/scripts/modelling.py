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

# Data science
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# Personal modules
sys.path.append( ".." )
from utils.generic import save_img

#################################################
#     splitting_dataset
#################################################
def model( X, Y ):
    
    # Using k-Fold cross-validation for train/test splitting
    print( "Chosen algorithm for train/test sets splitting:", cl( "k-Fold cross-validation", "green" ) )
    num_folds = 5 # 3-5-10
    kfold = KFold( n_splits = num_folds, random_state = 7, shuffle = True )
    print()
    
    # Applying the model
    model = LogisticRegression( solver = "lbfgs", max_iter = 500 )
    
    # Evaluating the model
    print( "Performance metrics" )
    results = cross_val_score( model, X, Y, cv = kfold, scoring = "accuracy" )
    print( "- Accuracy: ", end = "" )
    print( cl( "%.3f%% +/- %.3f%%" % ( results.mean()*100.0, results.std()*100.0 ), "yellow" ) )

    results = cross_val_score( model, X, Y, cv = kfold, scoring = "neg_log_loss" )
    print( "- Negative log-loss: ", end = "" )
    print( cl( "%.3f +/- %.3f" % ( results.mean(), results.std() ), "yellow" ) )

    results = cross_val_score( model, X, Y, cv = kfold, scoring = "roc_auc" )
    print( "- Area under the ROC curve: ", end = "" )
    print( cl( "%.3f +/- %.3f" % ( results.mean(), results.std() ), "yellow" ) )

    print( "- Saving confusion matrix" )
    predicted = cross_val_predict( model, X, Y, cv = kfold )
    matrix = confusion_matrix( Y, predicted )
    df_cm = pd.DataFrame( matrix )
    plt.figure( figsize = ( 10,7 ) )
    sn.heatmap( df_cm, annot = True, cmap = "YlOrRd", fmt = "d" )
    save_img( "confusion_matrix", model_path )

#################################################
#     Main
#################################################
def main():
    
    # Global variables
    global model_path
    model_path = "../../img/modelling"
    
    # Loading the dataset
    data = pd.read_csv( "../../data/processed_data.csv" )
    array = data.values
    X = array[ :, 0:15 ]
    Y = array[ :, 15 ]

    # Performing the modelling
    model( X, Y )

if __name__ == "__main__":
    main()    