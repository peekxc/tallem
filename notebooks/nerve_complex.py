
import numpy as np 
from tallem.dimred import *



X = np.random.uniform(size=(15, 2), low = 0, high = 1.0)
G = neighborhood_graph(X, k = 5).A
knn_graph(X, k = 3)
import matplotlib.pyplot as plt
plt.scatter(*X.T)
# for 
plt.scatter(*G.T)