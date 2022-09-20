#!/bin/bash

# $1 -> processing / modelling

# Processing the dataset
if [ "$1" == "" ] || [ "$1" == "processing" ] ; then
    echo "##################################################"
    echo "#     Data preparation"
    echo "##################################################"
    echo ""

    ./data_preparation.py \
    --data="../data/sonar.all-data.csv" \
    --debugging="off" \
    --n_of_features=14
fi

if [ "$1" == "" ] ; then
    echo ""
fi

# Applying the model
if [ "$1" == "" ] || [ "$1" == "modelling" ] ; then
    echo "##################################################"
    echo "#     Modelling"
    echo "##################################################"
    echo ""

    ./modelling.py \
    --data="../data/processed_data.csv" \
    --n_of_folds=10 \
    --hyperparametrization="on"
fi