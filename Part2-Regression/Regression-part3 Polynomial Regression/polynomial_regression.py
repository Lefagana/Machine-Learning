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
x = dataset.iloc[:,:-1].values
#dataset  purchase column
y = dataset.iloc[:,3].values 