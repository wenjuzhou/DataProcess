import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LassoCV, MultiTaskLassoCV

import matplotlib.pyplot as plt
from matplotlib import collections, colors, transforms
import os

'''
# multiTaskLasso test
X_train = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
Y_train = np.array([[1,2,3],[4,4,4]])
X_test = np.array([[13,14,15,16],[17,18,19,20],[21,22,23,24]])


reg = MultiTaskLassoCV()

reg.fit(X_train,Y_train.T)
y_pred = reg.predict(X_test)
print(y_pred)
'''

# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Creates just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# Creates four polar axes, and accesses them through the returned array
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axes[0, 0].plot(x, y)
axes[1, 1].scatter(x, y)

# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

# Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

# Creates figure number 10 with a single subplot
# and clears it if it already exists.
fig, ax=plt.subplots(num=10, clear=True)


plt.show()
os.system("pasuse")