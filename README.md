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
- Skewness
- Correlation matrix
- Plotting each numeric and factor variable with the help of a loop.

## Modelling

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
- If using a log or Box-Cox transformation, don’t center the data first or do any operations that might make the data non-positive. Alternatively, use the Yeo-Johnson transformation so you don’t have to worry about this.
- One-hot or dummy encoding typically results in sparse data which many algorithms can operate efficiently on. If you standardize sparse data you will create dense data and you lose the computational efficiency. Consequently, it’s often preferred to standardize your numeric features before one-hot/dummy encode.
- If you are lumping infrequently occurring categories together, do so before one-hot/dummy encoding.
- Although you can perform dimension reduction procedures on categorical features, it is common to primarily do so on numeric features concerning feature engineering purposes.

```
house_rec <- recipe(SalePrice ~ ., data = house_train) %>%
  update_role(Id, new_role = "ID") %>% # define the ID as ID to exclude it form modelling
  step_log(SalePrice) %>% # dealing with the skewness of the target variable
  step_knnimpute(all_predictors(), neighbors = 3) %>% # remove the last NA’s with KNN
  step_nzv(all_nominal())  %>% # using the Near-Zero Variance filter
  step_BoxCox(all_numeric(),-all_outcomes()) %>% # dealing with the skewness of the other variables 
  step_center(all_numeric(), -all_outcomes()) %>% # centering numeric data
  step_scale(all_numeric(), -all_outcomes()) %>% # scaling Numeric Data
  step_pca(all_numeric(), -all_outcomes())%>% # reduction of variables
  step_other(all_nominal(), -all_outcomes(), threshold = 0.01) %>% #potentially pool infrequently occurring values into an "other" category
  step_dummy(all_nominal(), -all_outcomes()) # creating dummies
```

Afterwards, I used the prep and the juice function and started the workflow.
 
```
house_prep <- prep(house_rec)

juiced <- juice(house_prep)

wf <- workflow() %>%
  add_recipe(house_rec)

test_prep <- house_prep %>%
  bake(house_test)
```
 
As resampling method, I decided to do a quick k-fold cross validation instead of bootstrapping. 
The models

```
train_cv <- house_train %>%
  vfold_cv(v = 5, strata = "SalePrice")
```

I applied a Regularized Regression, Random Forests and XGBoost. For all these models I proceeded as followed:
1.    Define a parsnip model
2.    Define parameters using dials package
3.    Combine model and recipe using workflows package
4.    Tune the workflow using tune package
5.    Evaluate tuning results
6.    Select best model for prediction
7.    Predict on test data
8.    Finally, save the prediction on the test data

 In addition, I promise not to go too deep into the optimization of each model, as this would go beyond the scope of these projects.

### Glmnet:
1.-3. I decided to tune penalty and mixture with a filter of penalty <= .01
```
# 1) Define a parsnip model
glmnet_mod <- linear_reg(penalty = tune(),
                         mixture = tune()) %>%
  set_engine("glmnet")
  
# 2) Define parameters using dials package
para <- parameters(penalty(),mixture())
glmnet_tune_grid <- grid_regular(para,
                                 filter = penalty <= .01,
                                 levels = 20
                                 )

# 3) Combine model and recipe using workflows package
glmnet_workflow <- 
    workflow() %>% 
    add_recipe(house_rec) %>% 
    add_model(glmnet_mod)
```
4.
```
# 4) Tune the workflow using tune package
tictoc::tic()
glmnet_tuned_results <- tune_grid(
  glmnet_workflow,
  resamples = ames_vfold,
  grid = glmnet_tune_grid,
  metrics = metric_set(rmse),
  control = control_grid()
)
tictoc::toc()
```
5.-6. I used RMSE to evaluate each model
```
# 5) Evaluate tuning results
show_best(glmnet_tuned_results, "rmse", n = 10)

# 6) Select best model for, e.g., prediction
glmnet_param_best <- select_best(glmnet_tuned_results, metric = "rmse")
glmnet_model_best <- finalize_model(glmnet_mod, glmnet_param_best)
glmnet_model_finalfit <- fit(glmnet_model_best, SalePrice ~ ., data = juiced)
```
7. Prediction of target variable test data
```
# 7) Predict on test data
test_prep <- house_prep %>%
  bake(house_test)

glmnet_preds <- 
    predict(glmnet_model_finalfit, new_data = test_prep) %>% 
    transmute(SalePrice = exp(.pred)) %>% 
    bind_cols(select(house_test, Id), .)
head(glmnet_preds)
```

### Random Forest:
1.-3. I decided to tune mtry and min_n
```
# 1) Define a parsnip model
rf_mod <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger")
  
 # 2) Is in this case not necessary
 
 # 3) Combine model and recipe using workflows package
rf_workflow <- 
    workflow() %>% 
    add_recipe(house_rec) %>% 
    add_model(rf_mod)
```
To go more into detail I recommend Bradley Boehmke and Brandon Greenwell’s tuning strategies for Random Forests.

4.
```
# 4) Tune the workflow using tune package
library(ranger)
tictoc::tic()
rf_tuned_results <- tune_grid(
  rf_workflow,
  resamples = ames_vfold,
  grid = 20
  )
tictoc::toc()
```
5.-6. I used RMSE to evaluate each model
```
# 5) Evaluate tuning results
show_best(rf_tuned_results, "rmse", n = 10)

# 6) Select best model for, e.g., prediction
rf_param_best <- select_best(rf_tuned_results, metric = "rmse")
rf_model_best <- finalize_model(rf_mod, rf_param_best)
rf_model_finalfit <- fit(rf_model_best, SalePrice ~ ., data = juiced)
```
7. Prediction of target variable using test data
```
# 7) Predict on test data
test_prep <- house_prep %>%
  bake(house_test)

rf_preds <- 
    predict(rf_model_finalfit, new_data = test_prep) %>% 
    transmute(SalePrice = exp(.pred)) %>% 
    bind_cols(select(house_test, Id), .)
```
