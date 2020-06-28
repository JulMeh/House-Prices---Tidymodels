# Predicting House Prices Tidymodels

## Introduction

In this document I am going to predict the selling price of houses. To do so, I  use data from Kaggle, do an EDA, do some transformations and finally use Advanced regression techniques to create a model for the prediction.

This project is based on the Kaggle competition "[House prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description)". For this reason I will also take over the general conditions of this competition:

- Competition Description: Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

- Acknowledgments: The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 

- Goal: It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

- Metric: Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

Moreover, I try to put a focus on the use of tidyverse packages, an appealing visualization of the data, and to create a readable html file. 

## Setup

During the import of the data I already defined some data types and some levels of Ordinal data. Afterwards I changed NA values, which were described in the text file. To finish the setup I renamed a few more variables and merged test and training data. 
 
Parts of my EDA are:
-      Skewness
-      Correlation matrix
-      Plotting each numeric and factor variable with the help of a loop.


In the following I made some final adjustments and split the data into test and train data.
Starting with tidymodels:
First, I started with a recipe. I built up this recipe based on the steps of Hands-On Machine Learning with R (Bradley Boehmke and Brandon Greenwell):
1.    Filter out zero or near-zero variance features.
2.    Perform imputation if required.
3.    Normalize to resolve numeric feature skewness.
4.    Standardize (center and scale) numeric features.
5.    Perform dimension reduction (e.g., PCA) on numeric features.
6.    One-hot or dummy encode categorical features.

Moreover, they give some general suggestions which should be considered:
•         If using a log or Box-Cox transformation, don’t center the data first or do any operations that might make the data non-positive. Alternatively, use the Yeo-Johnson transformation so you don’t have to worry about this.
•         One-hot or dummy encoding typically results in sparse data which many algorithms can operate efficiently on. If you standardize sparse data you will create dense data and you lose the computational efficiency. Consequently, it’s often preferred to standardize your numeric features before one-hot/dummy encode.
•         If you are lumping infrequently occurring categories together, do so before one-hot/dummy encoding.
•         Although you can perform dimension reduction procedures on categorical features, it is common to primarily do so on numeric features concerning feature engineering purposes.
