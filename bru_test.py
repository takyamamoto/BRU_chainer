# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:42:36 2019

@author: user
"""
import numpy as np
import chainer.functions as F
from BRU import eru
import matplotlib.pyplot as plt
x = np.arange(-10, 10, 0.1)
#y = F.elu(x, alpha=1.)
y = eru(x, r = 2)
#print(y.data)
plt.plot(y.data)
