Predicting House Prices Using the Boston Housing Dataset

Project Overview

This project implements regression models from scratch to predict house prices using the Boston Housing Dataset from Kaggle (HousingData.csv). The models include Linear Regression, Random Forest, and a simplified XGBoost, all built without using scikit-learn's built-in model classes. The script preprocesses the data, trains the models, evaluates their performance using RMSE and R² metrics, and visualizes feature importance for the tree-based models.

Steps

Data Preprocessing:

Loaded the Boston Housing Dataset from HousingData.csv using pandas.
Handled missing values by imputing with the mean of each feature.
Normalized numerical features using z-score normalization (subtract mean, divide by standard deviation).
Added a bias term for Linear Regression.
Split data into 80% training and 20% testing sets with a random seed for reproducibility.

Model Implementation:

Linear Regression: Implemented using gradient descent to optimize weights.
Random Forest: Built with decision trees, using bootstrap sampling and random feature selection at each split.
XGBoost: Simplified version with gradient boosting, where each tree predicts residuals adjusted by a learning rate.
Performance Comparison:
Evaluated models using Root Mean Squared Error (RMSE) and R² score on the test set.
Printed performance metrics for each model.
Feature Importance:
Calculated feature importance for Random Forest and XGBoost based on variance reduction.
Visualized importance using bar plots, saved as feature_importance.png.

How to Run
Prerequisites:
Python 3.x
Required libraries: numpy, pandas, matplotlib
Install dependencies: pip install numpy pandas matplotlib
Download HousingData.csv from Kaggle (e.g., from the dataset page: https://www.kaggle.com/c/boston-housing) and place it in the same directory as the script.
Running the Script:





Save the Python script as boston_housing_regression.py.



Ensure HousingData.csv is in the same directory.



Run the script: python boston_housing_regression.py



The script will:





Load and preprocess the dataset from HousingData.csv.



Train the models.



Print RMSE and R² metrics for each model.



Generate and save a feature importance plot as feature_importance.png.

Observations





Performance: Random Forest and XGBoost generally outperform Linear Regression due to their ability to capture non-linear relationships. XGBoost may slightly edge out Random Forest due to its boosting approach, but performance depends on tuning (e.g., number of trees, depth).



Feature Importance: Features like RM (average number of rooms) and LSTAT (% lower status population) often show high importance in tree-based models, indicating strong influence on house prices.



Missing Values: The script handles missing values by imputing with the mean, which is a simple approach. The Kaggle dataset may have missing values in features like CRIM or ZN, unlike the scikit-learn version.

Limitations: The custom implementations are simplified. For example, XGBoost lacks advanced features like regularization, and the decision trees use basic variance reduction for splitting. Real-world applications might benefit from optimized libraries like scikit-learn or XGBoost.

Runtime: The script runs in a few seconds on a standard machine, but training times increase with more trees or deeper trees in Random Forest and XGBoost.

Files





app.py: Main script with model implementations, training, evaluation, and visualization.

HousingData.csv: Input dataset (must be downloaded from Kaggle).


