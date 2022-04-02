# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:05:05 2022

@author: MALAH-LG
"""

#importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values 
