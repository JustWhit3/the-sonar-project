#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:21:00 2022
Author: Gianluca Bianco
"""

#################################################
#     Libraries
#################################################

# STD modules
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Personal modules
sys.path.append( ".." )
from utils.generic import save_img

#################################################
#     Main
#################################################
def main():
    data = pd.read_csv( "../../data/sonar.all-data.csv" )
    #print( dataset )
    
    # Printing histograms of each column
    data.hist()
    plt.rcParams[ "figure.figsize" ] = [ 16, 16 ]
    save_img( "histograms", "../../img/utility" )

if __name__ == "__main__":
    main()