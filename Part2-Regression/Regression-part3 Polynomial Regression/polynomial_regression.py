# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:08:18 2022

@author: MALAH-LG
"""

#importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
#dataset without purchase column
x = dataset.iloc[:,1:2].values
#dataset  purchase column
y = dataset.iloc[:,2].values 

##We dont need this section on this tut because we dont have enough info
#Splitting dataset into training set and test set
'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
'''
##also no need o this stage too
#feature Scaling
##

## comparing the linear and Polynomial Regression

# fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4 )
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualising the Linear Regression Results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Visualising the Polynomial Regression Results
'''x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))'''
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])
# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))