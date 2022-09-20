# The Sonar Project

Application of a machine learning classification algorithm to the Sonar dataset. Project for the course "Applied Machine Learning (Basic)" at PhD in Physics.

## Table of contents

- [Introduction](#introduction)
- [Software setup and run](#software-setup-and-run)
  - [Setup](#setup)
  - [Run](#run)
- [Data preprocessing](#data-preprocessing)
- [Modelling](#modelling)

## Introduction

This project is related to the course [Applied Machine Learning (Basic)](https://www.unibo.it/it/didattica/insegnamenti/insegnamento/2020/455026) at PhD in Physics. It consists of the application of machine learning to classify the [Sonar](https://www.kaggle.com/datasets/ypzhangsam/sonaralldata) data and discriminate between *rocks* and *minerals*.

Since this project is related to the basic part of the course, only basic machine learning algorithms for binary classification will be used and neural networks will not be considered.

## Software setup and run

### Setup

1) Download and unzip the repository.

2) Once the repository is downloaded and unzipped, `cd` into it and enter:

```Bash
source setup.sh
```

If it is the first time you run it, a virtual environment `venv` will be created and activated and prerequisites Python modules will be installed. From the second time on, the script will simply activate the virtual environment.

> :warning: be sure to have installed the `virtualenv` package with `pip install virtualenv` command.

3) Download the dataset from [here](https://www.kaggle.com/datasets/ypzhangsam/sonaralldata) and move it into the `data` directory:

```Bash
mkdir -p data
mv path/to/dataset data
```

### Run

First of all `cd` the `src` directory.

To run the data preprocessing part:

```Bash
./all_analysis.sh processing
```

To run the modelling part:

```Bash
./all_analysis.sh modelling
```

To run the entire analysis:

```Bash
./all_analysis.sh
```

## Data preprocessing

Data are distributed with 60 columns plus one containing only the labels to be used for classification. All them lie in the range between 0 and 1.

These procedures have been applied, in the following order, for data preprocessing:

- **Feature selection**: through the [`SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) algorithm which select features according to the k-highest scores. From this step it has been realized that only 14/60 features are considered really important.
- **Data standardization**: through the [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) which standardize feature by removing the mean and scaling to unit variance.
- **Data normalization**: through the [`Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) algorithm, which normalize samples individually to unit norm.

Some control plots used for feature exploration have been produced:

<img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/utility/histograms.png"  width = "450">

## Modelling
