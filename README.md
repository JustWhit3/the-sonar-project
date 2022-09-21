<p align="center"><img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/logo.svg" height=220></p>

<h3 align="center">Application of machine learning to classify to the Sonar data. </h3>
<p align="center">
  <img title="v0.0" alt="v0.0" src="https://img.shields.io/badge/version-v0.0-informational?style=flat-square">
  <img title="MIT License" alt="license" src="https://img.shields.io/badge/license-MIT-informational?style=flat-square">
	<img title="Python 3.10" alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-informational?style=flat-square"><br/>
	<img title="Code size" alt="code size" src="https://img.shields.io/github/languages/code-size/JustWhit3/the-sonar-project?color=red">
	<img title="Repo size" alt="repo size" src="https://img.shields.io/github/repo-size/JustWhit3/the-sonar-project?color=red">
	<img title="Lines of code" alt="total lines" src="https://img.shields.io/tokei/lines/github/JustWhit3/the-sonar-project?color=red"></br>
  <img title="codeq" alt="codeq" src="https://github.com/JustWhit3/the-sonar-project/actions/workflows/codeql-analysis.yml/badge.svg">
</p>

***

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

Data are distributed with 60 columns plus one containing only the labels to be used for classification. All them lie in the range between 0 and 1. First 6 columns are printed as an example:

```txt
         F0      F1      F2      F3      F4      F5
0    0.0200  0.0371  0.0428  0.0207  0.0954  0.0986
1    0.0453  0.0523  0.0843  0.0689  0.1183  0.2583
2    0.0262  0.0582  0.1099  0.1083  0.0974  0.2280
3    0.0100  0.0171  0.0623  0.0205  0.0205  0.0368
4    0.0762  0.0666  0.0481  0.0394  0.0590  0.0649
..      ...     ...     ...     ...     ...     ...
203  0.0187  0.0346  0.0168  0.0177  0.0393  0.1630
204  0.0323  0.0101  0.0298  0.0564  0.0760  0.0958
205  0.0522  0.0437  0.0180  0.0292  0.0351  0.1171
206  0.0303  0.0353  0.0490  0.0608  0.0167  0.1354
207  0.0260  0.0363  0.0136  0.0272  0.0214  0.0338
```

The following procedures have been applied, in this order, for data preprocessing:

- **Feature selection**: through the [`SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) algorithm which select features according to the k-highest scores. From this step it has been realized that only 14/60 features are considered really important.
- **Data standardization**: through the [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) which standardize feature by removing the mean and scaling to unit variance.
- **Data normalization**: through the [`Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) algorithm, which normalize samples individually to unit norm.

Some control plots used for feature exploration have then been produced after data manipulation:

<p align="center"><b>Histograms</b></br></br><img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/utility/histograms.png" width = "650"></p>

<p align="center"><b>Density Plots</b></br></br><img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/utility/density.png" width = "650"></p>

<p align="center"><b>Scatter Matrix</b></br></br><img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/utility/scatter_matrix.png" width = "650"></p>

<p align="center"><b>Box Plots</b></br></br><img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/utility/box.png" width = "650"></p>

<p align="center"><b>Correlation Matrix</b></br></br><img src="https://github.com/JustWhit3/the-sonar-project/blob/main/img/utility/correlation.png" width = "650"></p>

## Modelling

Work in progress...