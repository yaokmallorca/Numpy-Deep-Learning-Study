import numpy as np
from utils import *

a = np.arange(0, 96).reshape(2,4,4,3)
print(a.shape)
"""
w = np.random.randint(5, size=(3,3,3,5))
fr, fc, s, d = 3, 3, 1, 1

a_col, p = im2col(a, w.shape, (1,1), 1, dilation=1)
print("padding: ", p)
print(a_col)
print(a_col.shape)
"""

atrous = dilate(a, 2)
print(atrous.shape)