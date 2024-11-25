# Exploring determinant factors influencing muscle quality and sarcopenia in Bilbao's older adult population through machine learning: A comprehensive analysis approach.

This repository contains all of the code developed for working with eVida research group at the Universidad de Deusto, covering the above research article.<br>

# Repository Structure

[code](https://github.com/dmdequin/evida/tree/main/code): This directory contains all of the Jupyter notebooks and Python scripts used for this research.

[data](https://github.com/dmdequin/evida/tree/main/data): Contains all of the datasets used and subsets created over the course of the project.
- [sarco](https://github.com/dmdequin/evida/tree/main/data/sarco) - all of the sarcopenia datasets.
- [mqi](https://github.com/dmdequin/evida/tree/main/data/mqi) - all of the MQI datasets.
- Mayores_Completa_masde60años.csv - original dataset converted to csv.
- Mayores_Completa_masde60años.xlsx - original dataset.
- datos_modificados.csv - original dadtaset modified, with missing values or errors filled in.
- datos_puros.csv - original dataset modified, with rows removed that contained missing values or errors.

[notes](https://github.com/dmdequin/evida/tree/main/notes): Contains two markdown files: <br>
- journal.md contains daily and weekly progress notes, to track decision making and important progress details.
- meeting_notes.md contains agenda of tasks to complete.

[plots](https://github.com/dmdequin/evida/tree/main/plots): Each folder contains all plots and exported tables created while investigating the corresponding condition.
- [mqi](https://github.com/dmdequin/evida/tree/main/plots/mqi):
- [sarco](https://github.com/dmdequin/evida/tree/main/plots/sarco):

[results](https://github.com/dmdequin/evida/tree/main/results): Contains the results from each stage of classification experiments for both MQI and sarcopenia.
- [mqi](https://github.com/dmdequin/evida/tree/main/results/mqi):
- [sarco](https://github.com/dmdequin/evida/tree/main/results/sarco):


# To re-run this project:

requirements:



## Step 1: Create clean data
Import the original csv file and clean the data appropriately. Running the notebook ```data_cleaning.ipynb``` in its entirety will create two files: datos_modificados.csv and datos_puros.csv.

## Step 2: Prepare features for MQI investigation
```mqi_feature_preparation.ipynb```

## Step 3: Feature selection
```mqi_feature_selection.ipynb```
```sarcopenia_feature_selection.ipynb```

## Step 4: Classification
```mqi_classification.ipynb```
```sarcopenia_classification.ipynb```
