# ROCK-VS-STONE-PREDICTION-
 SONAR ROCK VS RUN PREDICTION IN MACHINE LEARNING USING LOGISTIC REGRESSION 


 This machine learning project uses Logistic Regression to classify objects detected by sonar signals as either rock or mine. The model is trained on the well-known Sonar dataset from the UCI Machine Learning Repository.

# Objective
Predict whether a sonar signal reflects from a rock or a mine.

Use logistic regression for binary classification.

Evaluate performance using accuracy and confusion matrix.

# Dataset Overview
Source: UCI Machine Learning Repository - Sonar Dataset

Instances: 208 samples

Features: 60 numeric attributes (energy of sonar signals in various frequency bands)

Target Variable:

R → Rock

M → Mine

# Technologies Used
Python

Pandas & NumPy

Matplotlib & Seaborn

scikit-learn (Logistic Regression, Train-Test Split, Metrics)

# Project Workflow
Import Libraries

pandas, numpy, matplotlib, seaborn, sklearn.

Load Dataset

Loaded CSV data using pandas.

Preprocess

Encode target labels (R, M) to binary values (0 and 1).

Feature scaling (if necessary).

Split Dataset

80% training, 20% testing using train_test_split.

Model Training

Logistic Regression using sklearn.linear_model.LogisticRegression.

Model Evaluation

Accuracy Score

Confusion Matrix

Classification Report

Prediction

Model predicts whether new sonar data points are rock or mine.
