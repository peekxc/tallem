

# from numba import config 
# config.THREADING_LAYER = 'safe'

from tallem.syevr import numba_dsyevr

import numpy as np 
from scipy.spatial.distance import squareform, pdist, cdist
from tallem.dimred import *

a = np.random.uniform(size=(3,3))
D = squareform(pdist(a))**2
D = -0.5*(D - average_rows(D) - average_cols(D).T + np.mean(D))

numba_dsyevr(D, 1, D.shape[0], 0)
np.linalg.eigh(D)

# Benchmark
import time 
a = np.random.uniform(size=(1500,3))
D = squareform(pdist(a))**2
D = -0.5*(D - average_rows(D) - average_cols(D).T + np.mean(D))

tic = time.perf_counter()
w = np.linalg.eigh(D)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
cc = numba_dsyevr(D, D.shape[0]-2, D.shape[0], 0)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")


cmds(D, D.shape[0], coords=False)

# From: https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyevr_ex.f.htm
x = np.array([
	[ 0.67, -0.20,  0.19, -1.06,  0.46],
	[-0.20,  3.82, -0.13,  1.06, -0.48],
	[ 0.19, -0.13,  3.27,  0.11,  1.10],
	[-1.06,  1.06,  0.11,  5.86, -0.98],
	[ 0.46, -0.48,  1.10, -0.98,  3.54]
], dtype=np.float64)
#  -0.98  -0.01  -0.08
# *   0.01   0.02  -0.93
# *   0.04  -0.69  -0.07
# *  -0.18   0.19   0.31
# *   0.07   0.69  -0.13
z = numba_dsyevr(x, 1, x.shape[0]-1, 0)

from tallem.dimred import cmds
evals, evecs = cmds(x, a.shape[0], coords=False)

print(evals)
print(z[0])
# 0.43   2.14   3.37
# %% 
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))	
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem/src/tallem")))	


#%% 
import numpy as np
n = 15
grid_x, grid_y = np.meshgrid(range(n), range(n))
X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)

X += np.random.uniform(size=X.shape, low=-0.05*n, high=0.05*n)

from src.tallem.distance import dist, subset_dist
from src.tallem.dimred import landmark_mds, cmds
from src.tallem.samplers import landmarks
from sklearn.manifold import MDS

import matplotlib.pyplot as plt 
plt.scatter(*X.T)
plt.scatter(*cmds(X).T)
plt.scatter(*cmds(dist(X, as_matrix=True)**2).T)
plt.scatter(*MDS(n_components=2, metric=True).fit_transform(X).T)

ind, radii = landmarks(X, 3)
Z = landmark_mds(dist(X, as_matrix=True)**2, ind, d = 2, normalize=False)
plt.scatter(*Z.T)

#%% 


# X[ind,:] == subset_dist(X, ind) 
# dist(subset_dist(X, ind)) == subset_dist(dist(X), ind)
# dist(subset_dist(X, ind), as_matrix=True) == subset_dist(dist(X, as_matrix=True), ind)


from sklearn.decomposition import PCA
plt.scatter(*PCA(n_components=2, svd_solver='full').fit_transform(X).T)



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
		X := pairwise distances, distance matrix, or point cloud.
		d := target dimension of output coordinatization 
		metric := if 'X' is not a distance matrix, the metric distance to compute on 'X'. Ignored otherwise. 
	'''
	D = dist(X, as_matrix=True, metric=metric)**2 if not(is_distance_matrix(X)) else X
	assert is_distance_matrix(D), "Must be a distance matrix."
	n = D.shape[0]
	H = centering_matrix(n)
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
import numpy as np
from numpy.typing import ArrayLike
from src.tallem.samplers import landmarks
from src.tallem.distance import dist

# X = np.random.normal(size=(100,5))


## Do maxmin to obtain landmarks, then apply classical MDS on landmarks
ind, radii = landmarks(X, 15)


X1 = landmark_mds(X, 2, ind, normalize=True)*2
X2 = landmark_mds(dist(X, as_matrix=True), 2, ind, normalize=True)

import matplotlib.pyplot as plt
plt.scatter(*X1.T, color='red')
plt.scatter(*X2.T, color='blue')

from scipy.stats import spearmanr
spearmanr(dist(X1), dist(X2))

from scipy.linalg import orthogonal_procrustes
R, sca = orthogonal_procrustes(X1, X2)

from src.tallem.alignment import opa
opa(X1, X2)

# L-ISOMAP: on the geodesic distance 
# Highly nonlinear data sets can be successfully linearised in this way, for example, a sample of images of a face
# under varying pose and lighting conditions, or a sample of images of a hand as the fingers and wrist movement
from src.tallem.dimred import neighborhood_graph
from scipy.sparse import csgraph
from scipy.linalg import eigh
from typing import *
from numpy.typing import ArrayLike

def landmark_isomap(X: ArrayLike, d: int = 2, subset: Union[str, ArrayLike] = "maxmin15", normalize=False):
	
	## Compute the neighborhood graph, the distances from every point to every landmark, and the landmark distance matrix
	G = neighborhood_graph(X, k = 15)
	S = csgraph.dijkstra(G, directed=False, indices=subset, return_predecessors=False)# Replaces Del?
	D = S[:,subset]

	## Apply classical MDS to landmark distance matrix
	n = D.shape[0]
	H = centering_matrix(n)
	B = -0.5 * H @ D @ H
	w, v = eigh(B, subset_by_index=list(range(n-d, n)))

	## Interpolate the lower-dimension points using the landmarks
	mean_landmark = np.mean(D, axis = 1).reshape((D.shape[0],1))
	L_pseudo = v/np.sqrt(w)
	Y = (-0.5*(L_pseudo.T @ (S - mean_landmark))).T 

	## Normalize using PCA, if requested
	if (normalize):
		Y_hat = Y.T @ centering_matrix(Y.shape[0])
		_, U = np.linalg.eigh(Y_hat @ Y_hat.T) # Note: Y * Y.T == (k x k) matrix
		Y = (U.T @ Y_hat).T
	return(Y)

landmark_mds(X, d=2, subset=ind)
landmark_isomap(X, d = 2, subset = ind)

# TODO: use pynndescent for neighborhood graph generation 

del_mu = np.mean(Del, axis=1).reshape((15,1))

w = landmark.maxmin(dist(X), -1.0, 500, True)




from src.tallem import landmark
from src.tallem.distance import dist

wut = landmark.do_parallel(10)
X = np.random.normal(size=(100,5))
w,v = landmark.cmds(dist(X, as_matrix=True))

import matplotlib.pyplot as plt
plt.scatter(*v[:,98:100].T)
