# Exploring determinant factors influencing muscle quality and sarcopenia in Bilbao's older adult population through machine learning: A comprehensive analysis approach.

This repository contains all of the code developed while working on the above research article within the eVida research group at the Universidad de Deusto.<br>

**ABSTRACT**
**BACKGROUND**: Sarcopenia and reduced muscle quality index have garnered special attention due to their prevalence among older individuals and the adverse effects they generate. Early detection of these geriatric pathologies holds significant potential, enabling the implementation of interventions that may slow or reverse their progression, thereby improving the individual's overall health and quality of life. In this context, artificial intelligence opens up new opportunities to identify the key identifying factors of these pathologies, thus facilitating earlier intervention and personalized treatment approaches.

**OBJECTIVES**: To investigate anthropomorphic, functional, and socioeconomic factors associated with muscle quality and sarcopenia using machine learning approaches and identify key determinant factors for their potential future integration into clinical practice.

**METHODS**: A total of 1253 older adults (89.5% women) with a mean age of 78.13 ± 5.78 voluntarily participated in this descriptive cross-sectional study, which examines determining factors in sarcopenia and MQI using machine learning techniques. Feature selection was completed using a variety of techniques and feature datasets were constructed according to feature selection. Three machine learning classification algorithms classified sarcopenia and MQI in each dataset, and the performance of classification models was compared.

**RESULTS**: The predictive models used in this study exhibited accuracy rates of 72.78% for MQI and 74.14% for sarcopenia, with the most successful algorithms being SVM and MLP. Key factors in predicting both conditions have been shown to be relative power, age, weight, and the 5STS. No single factor is sufficient to predict either condition, and by comprehensively considering all selected features, the study underscores the importance of a holistic approach in understanding and addressing sarcopenia and MQI among older adults.

**CONCLUSIONS**: Exploring the factors that affect sarcopenia and MQI in older adults, this study highlights that relative power, age, weight, and the 5STS are significant determinants. While considering these clinical markers and using a holistic approach, this can provide crucial information for designing personalized and effective interventions to promote healthy aging.

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


# Getting Started

## Prerequisites
- python 3.10.12

## Installation

1. Clone the repository: **`git clone https://github.com/dmdequin/sarcopenia_and_machine_learning.git`**
2. Navigate to the project directory: **`sarcopenia_and_machine_learning`**
3. Install dependencies: **`pip install -r requirements.txt`**

## To reproduce results:
### Step 1: Create clean data
Import the original csv file and clean the data appropriately. Run the notebook **```data_cleaning.ipynb```** in its entirety to create two files: datos_modificados.csv and datos_puros.csv.

### Step 2: Prepare features for MQI investigation
Run the following jupyter notebook:<br>
**```mqi_feature_preparation.ipynb```**

### Step 3: Feature selection
Run the following jupyter notebooks:<br>
**```mqi_feature_selection.ipynb```**<br>
**```sarcopenia_feature_selection.ipynb```**

### Step 4: Classification
Run the following jupyter notebooks:<br>
**```mqi_classification.ipynb```**<br>
**```sarcopenia_classification.ipynb```**


# License
Distributed under the MIT License. See [LICENSE.txt](https://github.com/dmdequin/sarcopenia_and_machine_learning/license.txt) for more information.

# Contact
Danielle Dequin - [LinkedIn](https://www.linkedin.com/in/danielle-dequin/)

Naiara Virto Castro - [LinkedIn](https://www.linkedin.com/in/naiara-v-05b420208/)
