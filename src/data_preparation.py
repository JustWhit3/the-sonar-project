#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:21:00 2022
Author: Gianluca Bianco
"""

#################################################
#     Libraries
#################################################

# Generic modules
import os
from emoji import emojize
from termcolor import colored as cl
import argparse as ap

# Data science
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, chi2

# Personal modules
from utils import save_img

#################################################
#     apply_operation
#################################################
def apply_operation( operation, data ):
    """
    Function used to apply a specific "operation" to the dataset "data" (ex. Normalizer, Standard Scaler etc...).

    Args:
        operation (sklearn.preprocessing): a specific operation to be applied.
        data (pandas.DataFrame): the dataset to which the operation is applied.

    Returns:
        pandas.DataFrame: the modified dataset.
    """
    
    nolabel_data = data.iloc[: , :-1]
    label = data[ "C" ]
    scaled_data = operation.fit_transform( nolabel_data )
    scaled_data_df = pd.DataFrame( scaled_data, index = data.index, columns = new_names )
    data = scaled_data_df.join( label )
    
    return data

#################################################
#     feature_selection
#################################################
def feature_selection( data, print_steps = False ):
    """
    Function used to perform feature selection on data.

    Args:
        data (pandas.DataFrame): the dataframe to be analyzed.
        print_steps (bool, optional): Choose if printing debugging steps or not. Defaults to False.

    Returns:
        pandas.DataFrame: The processed dataset.
    """

    # Variables
    label = data[ "C" ] 
    array = data.values
    X = array[ :, 0:60 ]
    Y = array[ :, 60 ]

    # Tests
    if print_steps == True:
        test = SelectKBest( score_func = chi2, k = 3 )
        fit = test.fit( X, Y )
        print( end = "\n\n" )
        print( fit.scores_ )
    
    # Feature selection
    X_new = SelectKBest( score_func = chi2, k = int( args.n_of_features ) ).fit_transform( X, Y )
    X_new_df = pd.DataFrame( X_new, index = data.index, columns = new_names )
    data = X_new_df.join( label )
    
    return data

#################################################
#     process_dataset
#################################################
def process_dataset( print_steps = False ):
    """
    Function used to load and preprocess the dataset.

    Args:
        print_steps (bool, optional): Choose if printing debugging steps or not. Defaults to False.

    Returns:
        pandas.DataFrame: the modified and loaded dataset.
    """

    # Loading the dataset
    print( "- Loading the dataset ", end = "" )
    data = pd.read_csv( args.data, names = names )
    data.rename( columns = { "F60": "C" }, inplace = True )
    print( emojize( ":check_mark_button:" ) )
    if print_steps == True:
        print( data )
        
    # Feature selection
    print( "- Performing feature selection ", end = "" )
    data = feature_selection( data, print_steps = print_steps )
    print( emojize( ":check_mark_button:" ) )
    if print_steps == True:
        print( data )
    
    # Standardizing the data
    print( "- Standardizing the dataset ", end = "" )
    data = apply_operation( StandardScaler(), data )
    print( emojize( ":check_mark_button:" ) )
    if print_steps == True:
        print( data )

    # Normalizing the data
    print( "- Normalizing the dataset ", end = "" )
    data = apply_operation( Normalizer(), data )
    print( emojize( ":check_mark_button:" ) )
    if print_steps == True:
        print( data )
    
    return data

#################################################
#     utility_plots
#################################################
def utility_plots( data ):
    """
    Function used to plot utility features of the "data" dataset. In particular: histograms, density plots, box plots, correlations plot and scatter matrix.

    Args:
        data (pandas.DataFrame): the dataset which features plots are required.
    """

    # Variables and constants
    dim = int( int( args.n_of_features ) / 3 )
    dimension = ( dim, dim )
    
    # Plots settings
    plt.rcParams[ "figure.figsize" ] = [ 16, 16 ]
    
    # Histograms
    print( "- Printing histograms of each column ", end = "" )
    data.hist()
    plt.tight_layout()
    save_img( "histograms", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Density plots
    print( "- Printing density plots ", end = "" )
    data.plot( kind = "density", subplots = True, layout = dimension, sharex = False )
    plt.tight_layout()
    save_img( "density", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Box plots
    print( "- Printing box plots ", end = "" )
    data.plot( kind = "box", layout = dimension, sharex = False, sharey = False )
    plt.tight_layout()
    save_img( "box", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Correlations
    print( "- Printing correlations plot ", end = "" )
    correlations = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot( 111 )
    cax = ax.matshow( correlations, vmin = -1, vmax = 1 )
    fig.colorbar( cax )
    ticks = np.arange( 0, 14, 1 )
    ax.set_xticks( ticks )
    ax.set_yticks( ticks )
    names = [ str( name ) for name in range( 0, 14 ) ]
    ax.set_xticklabels( names )
    ax.set_yticklabels( names )
    plt.tight_layout()
    save_img( "correlation", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Scatter matrix
    print( "- Printing scatter matrix ", end = "" )
    pd.plotting.scatter_matrix( data )
    plt.tight_layout()
    save_img( "scatter_matrix", utility_path )
    print( emojize( ":check_mark_button:" ) )

#################################################
#     Main
#################################################
def main():
    
    # Global variables
    global utility_path, names, new_names
    names = [ "F" + str( name ) for name in range( 0, 61 ) ]
    new_names = [ "F" + str( name ) for name in range( 0, int( args.n_of_features ) ) ]
    utility_path = "../img/utility"
    
    # Variables
    data_path = "../data"
    
    # Processing the dataset
    print( cl( "Dataset operations:", "green" ) )
    if args.debugging == "on":
        data = process_dataset( print_steps = True )
    elif args.debugging == "off":
        data = process_dataset( print_steps = False )
    if not os.path.exists( data_path ):
        os.makedirs( data_path )
    data.to_csv( "{}/{}".format( data_path, "processed_data.csv" ) )
    print()
    
    # Printing utility plots
    print( cl( "Plotting utils:", "green" ) )
    utility_plots( data )
    print()
    
    # Extra information
    print( "Plots have been saved in:", cl( utility_path, "yellow" ) )
    print( "Processed data has been saved in:", cl( data_path, "yellow" ) )
    
if __name__ == "__main__":

    # Argument parser settings
    parser = ap.ArgumentParser( description = "Argument parser for data preparation." ) 
    parser.add_argument( "--data", default = "../data/sonar.all-data.csv", help = "Input dataset." )
    parser.add_argument( "--debugging", default = "off", help = "Set debugging option (on/off)." )
    parser.add_argument( "--n_of_features", default = 60, help = "Select the number of important features." )
    args = parser.parse_args()

    # Running the program
    main()