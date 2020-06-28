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
