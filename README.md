# Exploring determinant factors influencing muscle quality and sarcopenia in Bilbao's older adult population through machine learning: A comprehensive analysis approach.

This repository contains all of the code developed while working on the above research article within the eVida research group at the Universidad de Deusto.<br>

# Repository Structure

[code](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/code): This directory contains all of the Jupyter notebooks and Python scripts used for this research.

[data](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/data): Contains all of the datasets used and subsets created over the course of the project.
- [sarco](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/data/mqi) - all of the sarcopenia datasets.
- [mqi](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/data/sarco) - all of the muscle quality (MQI) datasets.
- datos_completos.csv - complete original dataset.
- datos_modificados.csv - modified complete dataset with missing values or errors filled in.
- datos_puros.csv - pure dataset with rows removed that contained missing values or errors.

[plots](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/plots): Each folder contains all plots and exported tables created while investigating the corresponding condition.
- [mqi](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/plots/mqi): All feature selection and classification plots for muscle quality.
- [sarco](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/plots/sarco): All feature selection and classification plots for sarcopenia.

[results](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/results): Contains the results from each stage of classification experiments for both MQI and sarcopenia.
- [mqi](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/results/mqi): Muscle quality research results.
- [sarco](https://github.com/dmdequin/sarcopenia_and_machine_learning/tree/master/results/sarco): Sarcopenia research results.


# Setup to run this project:

requirements:

- python



# To reproduce results: 
## Step 1: Create clean data
Import the original csv file and clean the data appropriately. Running the notebook ```data_cleaning.ipynb``` in its entirety will create two files: datos_modificados.csv and datos_puros.csv.

## Step 2: Prepare features for MQI investigation
Run the following jupyter notebook:<br>
```mqi_feature_preparation.ipynb```

## Step 3: Feature selection
Run the following jupyter notebooks:<br>
```mqi_feature_selection.ipynb```<br>
```sarcopenia_feature_selection.ipynb```

## Step 4: Classification
Run the following jupyter notebooks:<br>
```mqi_classification.ipynb```<br>
```sarcopenia_classification.ipynb```
