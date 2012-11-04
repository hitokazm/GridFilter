'''
Created on Oct 16, 2012

@author: hitokazu
'''

import numpy as np
from grid_filter_gl import *


init_window(100,100)

grid = np.array(np.mat('1 0; 0 1'), ndmin=2)

update_grid(grid)

n = 0

while n < 10000000000000: 
    draw_grid()
    n += 1
