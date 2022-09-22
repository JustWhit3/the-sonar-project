#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Sep 15 15:48:00 2022
Author: Gianluca Bianco
"""

#################################################
#     Libraries
#################################################

# STD modules
import os
import argparse as ap
import collections
from termcolor import colored as cl

# Data science modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import numpy as np

# Personal modules
from utils import load_model

#################################################
#     get_models
#################################################
def get_models( path ):
    """
    Function used to create a dict of models to be studied.

    Args:
        path (str): the path in which models are saved.

    Returns:
        dict: the dict of models.
    """
    
    # Variables
    models = {}

    # Loading models
    model_LR = load_model( "LogisticRegression", path )
    model_LDA = load_model( "LinearDiscriminantAnalysis", path )
    model_KNN = load_model( "KNeighborsClassifier", path )
    model_CART = load_model( "DecisionTreeClassifier", path )
    model_NB = load_model( "GaussianNB", path )
    model_SVM = load_model( "SVC", path )
    model_RFC = load_model( "RandomForestClassifier", path )

    # Appending modules to the list
    models[ "LR" ] = model_LR
    models[ "LDA" ] = model_LDA
    models[ "KNN" ] = model_KNN
    models[ "CART" ] = model_CART
    models[ "NB" ] = model_NB
    models[ "SVM" ] = model_SVM
    models[ "RFC" ] = model_RFC
    
    return models

#################################################
#     accuracy_on_test_set
#################################################
def accuracy_on_test_set( model , X_test, Y_test ):
    """
    Function used to retrieve accuracy on the test set of a model.

    Args:
        model (sklearn): the model.
        X_test (np.Array): array of test set of data.
        Y_test (np.Array): array of test set of labels.

    Returns:
        sklearn: the score on the test set.
    """
    
    Y_predicted = model.predict( X_test )
    
    return accuracy_score( Y_test, Y_predicted )

#################################################
#     get_score
#################################################
def get_score( data, models ):
    """
    Function used to get the final score on the test set of each model.

    Args:
        data (pd:DataFrame): the dataset.
        models(dict): dictionary of saved models.
    """
    
    # Variables
    counter = 0
    models_acc = collections.defaultdict( list )

    # Loading the dataset
    data = pd.read_csv( data )
    dim = data.shape[1] - 1
    array = data.values
    X = array[ :, 0:dim ]
    Y = array[ :, dim ]
    
    # Getting accuracies of each split for the test set
    kfold = ShuffleSplit( n_splits = 100, test_size = 0.33, random_state = 7 )
    for train, test in kfold.split( X, Y ):
        counter += 1
        X_test, Y_test = X[ test ], Y[ test ]
        X_train, Y_train = X[ train ], Y[ train ]
        
        for model_name, model in models.items():
            model.fit( X_train, Y_train )
            acc = accuracy_on_test_set( model , X_test, Y_test )
            models_acc[ model_name ].append( acc )
    
    # Getting final accuracies
    for model in models:
        model_name = cl( model[0:], "green" )
        print( "- {}: ".format( model_name ), round( np.array( models_acc[ model ] ).mean()*100.0, 3) )

#################################################
#     Main
#################################################
def main():
    
    # Loading models
    models = get_models( "../models" )

    # Loading the dataset
    print( "Accuracies on the test set:" )
    get_score( args.data, models )
    
if __name__ == "__main__":
    
    # Argument parser settings
    parser = ap.ArgumentParser( description = "Argument parser for data preparation." ) 
    parser.add_argument( "--data", default = "../data/processed_data.csv", help = "Input dataset." )
    args = parser.parse_args()

    # Running the program
    main()   