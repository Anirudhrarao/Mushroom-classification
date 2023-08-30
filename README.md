# Mushroom Classification Project

## Overview

This repository contains a machine learning project that classifies mushrooms as edible or poisonous based on various characteristics. It includes data ingestion, data transformation, model training, and prediction pipelines. Several machine learning models were evaluated, with XGBoost achieving the best performance.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data](#data)
- [Models](#models)
- [Results](#results)

## Project Structure
```bash
├── src/
│   ├── components/
│   │   ├── __init__.py       # Initialization file for the components package
│   │   ├── data_ingestion.py # Module for data ingestion
│   │   ├── data_preprocessing.py # Module for data preprocessing
│   │   ├── model.py          # Module for machine learning model
│   ├── pipelines/
│   │   ├── __init__.py       # Initialization file for the pipelines package
│   │   ├── training_pipeline.py # Module for training pipeline
│   │   ├── predicting_pipeline.py # Module for predicting pipeline
│   ├── __init__.py           # Initialization file for the src package
│   ├── logger.py             # Module for logging
│   ├── exceptions.py         # Module for custom exceptions
│   ├── utils.py              # Module for utility functions
```
## Data

The dataset used for this project is available in the `data` directory. It includes a CSV file named `mushroom_data.csv` containing the mushroom data.

**Dataset Source**: [Mushroom Data](https://github.com/Anirudhrarao/Mushroom-classification/blob/main/dataset/mushroom_data.csv)


## Models
- Logistic Regression
- XGBoost Classifier
- Decision Tree
- Random Forest Classifier
- Gradient Boosting Classifier
- Ada Boost Classifier
- Support Vector Classifier
- CatBoost Classifier

## Getting started
To set up this project on your local machine, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/Anirudhrarao/Mushroom-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mushroom-classification
    ```
3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:
    ```bash
    venv\Scripts\activate
    ```

5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

6. Training model:
    ```bash
    python src\pipelines\training_pipeline.py
    ```
## Results
The XGBoost Classifier achieved the ``highest accuracy`` in classifying mushrooms as edible or poisonous.



