# %% 
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))	
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem/src/tallem")))	

# %% 
import numpy as np
from src.tallem.distance import dist

X = np.random.normal(size=(10000,5))
D = dist(X, as_matrix=True)**2

from src.tallem.distance import dist, is_distance_matrix 
from numpy.typing import ArrayLike
from scipy.linalg import eigh

def cmds(X: ArrayLike, d: int = 2, metric="euclidean"):
	''' 
	Classical Multi-dimensional Scaling (CMDS) 
		X := distance matrix representing squared euclidean distances, or a point cloud 
		d := target dimension of output coordinatization 
		metric := if 'X' is not a distance matrix, the metric distance to compute on 'X'. Ignored otherwise. 
	'''
	if not(is_distance_matrix(X)): 
		X = dist(X, as_matrix=True, metric=metric)**2
	assert is_distance_matrix(X), "'X' must be a distance matrix."
	n = X.shape[0]
	H = -np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	B = -0.5 * H @ D @ H
	w, v = eigh(B, subset_by_index=list(range(n-d, n)))
	assert np.all(w > 0), "Largest k eigenvalues not all positive"
	return(np.sqrt(w) * v)

from src.tallem.landmark import landmarks_dist
landmarks_dist()


# "Negative eigenvalues of Bn signify that the original distance matrix Dn is non-Euclidean."
## MDS error: np.linalg.norm((Y @ Y.T) - B)

#%% 
from src.tallem import TALLEM

# %% 
from src.tallem import landmark
from src.tallem.distance import dist

w = landmark.maxmin(X.T, -1.0, 500, False)
w = landmark.maxmin(dist(X), -1.0, 500, True)
