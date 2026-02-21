# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 11:16:08 2026

@author: Troy
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 1000)
y1 = 2 * x * np.cos(2 * x) - (x - 2)**2
y2 = -4 + 6 * x - (x**2) - 4 * (x**3)

plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')

# Fix: Added quotes to label strings
plt.xlabel('x')
plt.ylabel('y')
plt.legend() # Added legend to differentiate lines
plt.show()