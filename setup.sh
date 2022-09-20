#!/bin/bash

# Creating the virtual environment (only once)
if [[ ! -d venv ]] ; then
    echo "Creating the virtual environment..."
    virtualenv venv -p python3.10
    echo ""
    echo "Activating the virtual environment..."
    source venv/bin/activate
    echo ""
    echo "Installing prerequisites..."
    pip install -r requirements.txt
    echo ""
else
    echo "Activating the virtual environment..."
    source venv/bin/activate
    echo "Done."
fi