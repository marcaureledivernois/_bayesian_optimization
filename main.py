import numpy as np
import pandas as pd

def costly_function(x):
    total = np.array([])
    for x_i in x:
        total = np.append(total, np.sum(np.exp(-(x_i - 5) ** 2)))
    return total + np.random.randn()

x = np.random.randn(5,2)
y = costly_function(x)
pd.DataFrame(data={'y':y, 'x0':x[:,0], 'x1':x[:,1]})