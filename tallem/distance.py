# %% Distance imports
import numpy as np
import numpy.typing as npt 
from scipy.spatial.distance import cdist
from itertools import combinations

# %% Distance definitions
def dist(x: npt.ArrayLike, y=None, pairwise=False, as_matrix=False):
	''' 
	Returns 
		(1) all pairwise distances in x
		(2) all distances between x and y, or 
		(3) individual pairwise distances between x and y.
		(4) the distance between x and y, if both are 1 dimensional vectors
	'''
	x = np.array(x)
	if y is None:
		if (as_matrix):
			#d = np.zeros((n, n))
			#d[np.triu_indices(n=n,k=1,m=n)] = dist(x)
			#d = d + d.T
			d = cdist(x, x)
		else:
			n = x.shape[0]
			d = np.array([np.linalg.norm(x[cc[0],:] - x[cc[1],:]) for cc in combinations(range(n), 2)])
	else:
		n = x.shape[0]
		y = np.array(y)
		if pairwise:
			if x.shape != y.shape: raise Exception("x and y must have same shape.")
			d = np.array([np.linalg.norm(x[ii,:] - y[ii,:]) for ii in range(n)])
		else:
			if x.ndim == 1 and y.ndim == 1: 
				return(np.linalg.norm(x - y))
			if x.ndim == 1: x = np.reshape(x, (1, len(x)))
			if y.ndim == 1: y = np.reshape(y, (1, len(y)))
			n = x.shape[0]
			m = y.shape[0]
			d = np.zeros((n,m))
			for ii in range(n):
				d[ii,:] = np.array([np.linalg.norm(x[ii,:] - y[i,:]) for i in range(m)])
	return(d)
# %%
