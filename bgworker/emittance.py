# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:14:57 2018

@author: shliu
"""

import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('twissparameters.txt')
fig = plt.figure()
fig.add_subplot(231)
plt.plot(data1[:, 0])
plt.ylabel('x rms emit')
fig.add_subplot(232)
plt.plot(data1[:, 1])
plt.ylabel('x beta')
fig.add_subplot(233)
plt.plot(data1[:, 3])
plt.ylabel('x alpha')
fig.add_subplot(234)
plt.plot(data1[:, 4])

plt.ylabel('y rms emit')
fig.add_subplot(235)
plt.plot(data1[:, 5])

plt.ylabel('y beta')
fig.add_subplot(236)
plt.plot(data1[:, 7])
plt.ylabel('y alpha')
plt.tight_layout()
plt.savefig('convergence_wang.png')