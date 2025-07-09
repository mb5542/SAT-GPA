# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:38:39 2025

@author: Burzynek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# import data
data = pd.read_csv(r'D:\Programy\Python312\Scripts\sat-gpa\high_school_sat_gpa.csv', sep=' ',usecols=['math_SAT','verb_SAT','high_GPA'])

data.head()
data.dtypes

# Scatter plot showing the relationship between math_SAT and high_GPA
plt.figure(figsize=(7,5))
plt.scatter(data['math_SAT'], data['high_GPA'], color='b')
plt.show()

# Linear regression model lr_math that will describe the relationship between math_SAT and high_GPA
lr_math = LinearRegression()
lr_math.fit(X = data['math_SAT'].values.reshape(-1,1), y = data['high_GPA'].values)


# Chart presenting regression line data overlaid on a scatter plot
x_min = data['math_SAT'].min()
x_max = data['math_SAT'].max()

plt.figure(figsize=(7,5))
plt.scatter(data['math_SAT'], data['high_GPA'], color='b')
plt.plot([x_min, x_max],lr_math.predict([[x_min],[x_max]]), color='r')
plt.show()


# Scatter plot showing the relationship between math_SAT and high_GPA
plt.figure(figsize=(7,5))
plt.scatter(data['verb_SAT'], data['high_GPA'], color='g')
plt.show()

# Linear regression model lr_math that will describe the relationship between math_SAT and high_GPA
lr_verb = LinearRegression()
lr_verb.fit(X = data['verb_SAT'].values.reshape(-1,1), y = data['high_GPA'].values)

# Chart presenting regression line data overlaid on a scatter plot
x_min = data['verb_SAT'].min()
x_max = data['verb_SAT'].max()

plt.figure(figsize=(7,5))
plt.scatter(data['verb_SAT'], data['high_GPA'], color='g')
plt.plot([x_min, x_max],lr_verb.predict([[x_min],[x_max]]), color='r')
plt.show()

# Linear regression model working on two input variables (math_SAT and verbal_SAT)

lr = LinearRegression()
lr.fit(data[['math_SAT','verb_SAT']].values, data['high_GPA'].values.reshape(-1,1))


# Predicting High_GPA for specific student
student_john = np.array([600,650]).reshape(1,2)
john_score = lr.predict(student_john).item()
print(f'Student John will receive a grade of : {john_score:.2f}')
