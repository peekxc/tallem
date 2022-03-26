
import numpy as np 
from tallem.dimred import *
from tallem.datasets import mobius_band


X, P = mobius_band()


knn_graph(X, k = 3)
import matplotlib.pyplot as plt
plt.scatter(*X.T)
# for 
plt.scatter(*G.T)