# Machine learning for Regression 
## 2.1 Car price prediction project 
use dataset for car prices from kaggle
MSRP stands for Manufacturer Suggested Retail Price
### Project Plan
* Prepare data and do EDA (Exploratory Data Analysis
* Use linear regression for predicting price
* Understand the internals of linear regression
* Evaluating the model with RMSE (Root Mean Square
* Feature Engineering i.e adding new features
* Regularization - solving problems in our model
* Using the model

## 2.2 Data Preparation
* first we clean the data
str.lower().str.replace(' ', '_')

## 2.3 Exploratory Data Analysis 
This is used to understand how the data looks like. To learn more about the data and the problem 

Matplotlib and Seaborn is used to visualize the disteibution of price 

%matplotlib inline- this is used to make the plot visible in jupyter notebook 
np.log1p([...]) is used to get rid of long-tail distribution. This makes the plot a normal distribution which is good for models

*also check for missing values

## 2.4 Setting Up the Validation Framework 
using pandas and numpy 
* split the dataset into three
Train 60%
validation 20%
Test 20%

* Afterwards the target data  must be deleted to prevent it from being used accidentally for training purposes

## 2.5 Linear Regression 
linear regression is a model used for predicting numbers so that the output of a model is a number
Therefore the formula for linear regression is

$g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

$g(x_i) = w_0 + \displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$

in code is represented as:
~~~~python 
def linear_regression(xi):
    n = len(xi)
    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred
~~~~

## 2.6 Linear Regression Vector Form
If we look at the $\displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$ part in the above equation, we know that this is nothing else but a vector-vector multiplication. Hence, we can rewrite the equation as $g(x_i) = w_0 + x_i^T \cdot w$

We need to assure that the result is shown on the untransformed scale by using the inverse function `exp()`. 

## 2.7 Training a linear regression model - normal equation 
ordinarily the solution to Xw = y does not exist
because X is a rectangular matrix of many rows and few columns, therefore it's difficult to find the inverse of X
* Normal equation
  
$w$ = $(X^TX)^{-1}X^Ty$

Where:

$X^TX$ is the Gram Matrix and it's inverse exists because it's a square matrix

## 2.8 Baseline Model for Car Price Prediction Project 
this where we actually solve the problem us

we first implement our Train linear_regression model using the features

then we plot the prediction together with the trained model to see if we are correct
if the prediction is not accurate, we will need a way to quantity how bad the model is using RMSE

## 2.9 RMSE - Root Mean Square Error
we use this to objectively evaluate the performance of our linear regression model

$$RMSE = \sqrt{ \frac{1}{m} \sum {(g(x_i) - y_i)^2}}$$

- $g(x_i)$ is the prediction
- $y_i$ is the actual
- $m$ is the number of observations in the dataset (i.e. cars)


# 2.10 Computing RMSE on Validation Data
Here we take the train-dataset,  train the linear regression model then we apply it to the Validation data and look at the RMSE on the Validation data

## 2.11 Simple Feature Engineering 
* This is used to improve our model by adding more Features

## 2.12 Categorical Variables 
* These are variables that are typically strings e.g make of car, model

* It's data types are objects

* we add these variables to our model to improve our model 

* we encode it by representing them with a bunch of binary columns e.g number of doors for instance


## 2.13 Regularization 
this is a way to solve the problem of duplicated columns
in the previous lesson we added more Categorical values which gave extremely large RMSE 
In this lesson we learn why this happens using Regularization 

* one possibility could be that there are duplicate features in our data
* This would render the inverse of (XTX) non-existent

* one way we could solve this is to add a small number (r) to the diagonal of the matrix
* It could also be that our data is not super clean

## 2.14 Tuning the model
in this lesson we learn how to determine the best regularisation 'r' parameter for our linear regression model 

*Then when we get the best r, we run on our Validation dataset to see if it gives a good score 

## 2.15 Using The Model 
Here we train the final model on the test dataset by coming the training data and validation data as a single training model 
* the we check the RMSE
* So it shouldn't be too different from what we got in the Validation dataset
