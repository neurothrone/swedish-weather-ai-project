# Sweden Temperature Prediction

A machine learning project that predicts temperatures in Sweden based on historical weather data from Nordic countries (
2015-2019).

## Data Source

The project uses weather data from Finland, Norway, and Sweden (2015-2019) sourced
from [Kaggle](https://www.kaggle.com/datasets/adamwurdits/finland-norway-and-sweden-weather-data-20152019).

## Features

- Data preprocessing and cleaning
- Temperature prediction using machine learning models
- Model performance evaluation
- Visualization of actual vs predicted temperatures

## Models

The project implements and compares two machine learning models:

- Linear Regression
- Random Forest Regressor

## Dependencies

- pandas
- matplotlib
- scikit-learn

## Usage

1. Ensure you have the required dependencies installed
2. Run the main script:
   ```
   python main.py
   ```
3. The script will:
    - Load and preprocess the weather data
    - Train prediction models
    - Display performance metrics
    - Generate a visualization comparing actual vs predicted temperatures

## Data Description

The dataset includes the following features:

- country: Finland, Norway, or Sweden
- date: Date of observation
- precipitation: Daily precipitation amount
- snow_depth: Depth of snow
- tavg: Average temperature
- tmax: Maximum temperature
- tmin: Minimum temperature