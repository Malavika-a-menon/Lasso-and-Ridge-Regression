# Lasso-and-Ridge-Regression
Regression model using Lasso and Ridge Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:/Users/Admin/Downloads/gender_classification_v7.csv')
data.head()


x = data.iloc[:,:-1] ## independent features
y = data.iloc[:,-1]  ## dependent features

#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1,5,10,20,30,35,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring = 'neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1,5,10,20,30,35,45,50,55,100]}
lasso_regressor = GridSearchCV(lasso,parameters,scoring = 'neg_mean_squared_error',cv=5)

lasso_regressor.fit(x,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
prediction_lasso = lasso_regressor.predict(x_test)
prediction_ridge=ridge_regressor.predict(x_test)


import seaborn as sns
sns.distplot(y_test-prediction_lasso)
import seaborn as sns
sns.distplot(y_test-prediction_ridge)
