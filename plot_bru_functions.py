# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:42:36 2019

@author: user
"""
import numpy as np
from eru import eru
from oru import oru
import matplotlib.pyplot as plt
import os

save_dir = "./figs/"
os.makedirs(save_dir, exist_ok=True)


x = np.arange(-5, 5, 0.1)

n = 3
fig, ax = plt.subplots(figsize=(4,4))
for i in range(n):
    y = eru(x, r=(i+1))
    plt.plot(x, y.data, label='E'+str(i+1)+"RU")

ax.set_title("ERU")
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.grid(True)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir+"ERU.png")
#plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(4,4))
for i in range(n):
    y = oru(x, r=(i+1))
    plt.plot(x, y.data, label='O'+str(i+1)+"RU")

ax.set_title("ORU")
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.grid(True)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir+"ORU.png")
#plt.show()
plt.close()


