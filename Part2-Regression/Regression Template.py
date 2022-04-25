# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:09:17 2022

@author: MALAH-LG
Regreesion Template
"""
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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


# fitting the Regression Model to the dataset
# Create Your Regressor Here    

# Predicting a new result with Linear Regression
y_predict = regressor.predict([[6.5]])

# Visualising the Regression Model Results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression Model Results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()