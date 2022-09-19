#!/bin/bash

# Creating the virtual environment (only once)
if [[ ! -d venv ]] ; then
    echo "Creating the virtual environment..."
    virtualenv venv -p python3.10
    echo ""
    echo "Installing prerequisites..."
    pip install -r requirements.txt
    echo ""
fi

# Activating the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate
echo "Done."