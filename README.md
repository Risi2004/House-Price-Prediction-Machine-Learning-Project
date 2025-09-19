# House-Price-Prediction-Machine-Learning-Project


## Overview
This project focuses on building a machine learning model to predict house prices in Boston using the provided dataset. The process involves data loading, exploratory data analysis, data splitting, model training, and evaluation. The model is built using the **XGBoost Regressor**, an efficient and scalable implementation of gradient boosting.

## Files in this Repository
- **House_Price_Prediction.ipynb**: A Jupyter Notebook containing all the code for the project, including data processing, model training, and evaluation.
- **Boston.csv**: The dataset used for training and testing the model. It includes various features related to housing, such as per capita crime rate, number of rooms, and property tax rate.

## Dataset
The `Boston.csv` dataset contains **506 rows** and **13 columns**.  
The dataset has no missing values, and the target variable for prediction is **`medv`** (Median value of owner-occupied homes in $1000s).

### Features
- **crim**: Per capita crime rate by town  
- **zn**: Proportion of residential land zoned for lots over 25,000 sq. ft.  
- **indus**: Proportion of non-retail business acres per town  
- **nox**: Nitrogen oxides concentration (parts per 10 million)  
- **rm**: Average number of rooms per dwelling  
- **age**: Proportion of owner-occupied units built before 1940  
- **dis**: Weighted distance to five Boston employment centers  
- **rad**: Index of accessibility to radial highways  
- **tax**: Full-value property tax rate per $10,000  
- **ptratio**: Pupil-teacher ratio by town  
- **black**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town  
- **lstat**: % of lower status of the population  
- **medv**: Median value of owner-occupied homes in $1000s (Target variable)  

## Methodology
The project follows these steps:
1. **Data Loading**: The dataset is loaded into a Pandas DataFrame.  
2. **Exploratory Data Analysis (EDA)**: Dataset shape, missing values, and statistical measures are checked.  
3. **Data Splitting**: The data is split into features (X) and the target variable (Y), and then further divided into training and testing sets.  
4. **Model Training**: An XGBoost Regressor model is trained on the training data.  
5. **Model Evaluation**: The model’s performance is evaluated using **R-squared error** and **Mean Absolute Error** on both training and test data.  

## Dependencies
The following Python libraries are required:
- **pandas**: For data manipulation and analysis.  
- **numpy**: For numerical operations.  
- **matplotlib**: For data visualization.  
- **seaborn**: For creating statistical graphics.  
- **scikit-learn**: For model selection and evaluation metrics.  
- **xgboost**: For the regression model.  
