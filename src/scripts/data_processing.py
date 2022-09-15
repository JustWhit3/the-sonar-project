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
import sys
from emoji import emojize
from termcolor import colored as cl

# Data science
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, chi2

# Personal modules
sys.path.append( ".." )
from utils.generic import save_img

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
    label = data[ "60" ]
    scaled_data = operation.fit_transform( nolabel_data )
    scaled_data_df = pd.DataFrame( scaled_data, index = data.index, columns = data.columns[ 0: -1 ] )
    data = scaled_data_df.join( label )
    
    return data

#################################################
#     feature_selection
#################################################
def feature_selection( data, print_steps = False ):

    # Variables
    label = data[ "60" ] 
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
    X_new = SelectKBest( score_func = chi2, k = n_of_features ).fit_transform( X, Y ) # TODO: change k
    X_new_df = pd.DataFrame( X_new, index = data.index, columns = range( n_of_features ) )
    data = X_new_df.join( label )
    
    return data

#################################################
#     process_dataset
#################################################
def process_dataset( print_steps = False ):
    """
    Function used to load and preprocess the dataset.

    Returns:
        pandas.DataFrame: the modified and loaded dataset.
        bool: Choose if printing debugging steps or not. Default False.
    """

    # Loading the dataset
    print( "- Loading the dataset ", end = "" )
    names = [ str( name ) for name in range( 0, 61 ) ]
    data = pd.read_csv( "../../data/sonar.all-data.csv", names = names )
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
        
    # Renaming the last column
    data.rename( columns = { "60": "Label" }, inplace = True )
    
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
    dimension = ( 4, 4 )
    
    # Histograms
    print( "- Printing histograms of each column ", end = "" )
    data.hist()
    plt.rcParams[ "figure.figsize" ] = [ 16, 16 ]
    save_img( "histograms", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Density plots
    print( "- Printing density plots ", end = "" )
    data.plot( kind = "density", subplots = True, layout = dimension, sharex = False )
    save_img( "density", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Box plots
    print( "- Printing box plots ", end = "" )
    data.plot( kind = "box", layout = dimension, sharex = False, sharey = False )
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
    save_img( "correlation", utility_path )
    print( emojize( ":check_mark_button:" ) )
    
    # Scatter matrix
    print( "- Printing scatter matrix ", end = "" )
    pd.plotting.scatter_matrix( data )
    save_img( "scatter_matrix", utility_path )
    print( emojize( ":check_mark_button:" ) )

#################################################
#     Main
#################################################
def main():
    
    # Global variables
    global utility_path, n_of_features
    utility_path = "../../img/utility"
    n_of_features = 14
    
    # Variables
    data_path = "../../data"
    
    # Processing the dataset
    print( cl( "Dataset operations:", "green" ) )
    data = process_dataset( print_steps = False )
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
    main()