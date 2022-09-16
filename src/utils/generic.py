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
        >>> save_img( "save_img_test", "../../img/tests" )
        >>> os.path.exists( "../../img/tests/save_img_test.png" )
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
#     Main
#################################################
if __name__ == "__main__":
    doctest.testmod()