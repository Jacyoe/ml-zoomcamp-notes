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

