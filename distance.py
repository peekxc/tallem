import numpy as np 
from scipy.spatial.distance import cdist


from itertools import combinations
def dist(x, y=None, pairwise=False, as_matrix=False):
	''' 
	Returns 
		(1) all pairwise distances in x
		(2) all distances between x and y, or 
		(3) individual pairwise distances between x and y.
	'''
	n = x.shape[0]
	if y is None:
		if (as_matrix):
			#d = np.zeros((n, n))
			#d[np.triu_indices(n=n,k=1,m=n)] = dist(x)
			#d = d + d.T
			d = cdist(x, x)
		else:
			d = np.array([np.linalg.norm(x[cc[0],:] - x[cc[1],:]) for cc in combinations(range(n), 2)])
	else:
		if pairwise:
			if x.shape != y.shape: raise Exception("x and y must have same shape.")
			d = np.array([np.linalg.norm(x[ii,:] - y[ii,:]) for ii in range(n)])
		else:
			m = y.shape[0]
			d = np.zeros((n,m))
			for ii in range(n):
				d[ii,:] = np.array([np.linalg.norm(x[ii,:] - y[i,:]) for i in range(m)])
	return(d)