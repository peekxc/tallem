# %% Need for interactive scripts 
import os 
os.chdir("/Users/mpiekenbrock/tallem")
%reload_ext autoreload
%autoreload 2

# %% Imports 
import numpy as np
from tallem.distance import dist
from tallem.isomap import isomap
from tallem.mds import classical_MDS

# %% Setup data 
X = np.random.uniform(size=(15,2))
Y = isomap(X, d = 2, k = 3)
x_mds = classical_MDS(dist(X, as_matrix=True))

# %% Plot local euclidean models
import matplotlib.pyplot as pyplot
pyplot.scatter(X[:,0], X[:,1])
pyplot.scatter(Y[:,0], Y[:,1])
pyplot.scatter(x_mds[:,0], x_mds[:,1])

# %% Verify MDS
from sklearn.manifold import MDS
metric_mds = MDS(n_components = 2, eps = 1e-14)
Y_fit = metric_mds.fit(X)
Y = Y_fit.embedding_
print(Y_fit.stress_)

pyplot.draw()
pyplot.scatter(X[:,0], X[:,1])
pyplot.scatter(Y[:,0], Y[:,1])
pyplot.show()

# %% Procrustes 
from tallem.procrustes import ord_procrustes
aligned = ord_procrustes(X, Y, transform=True)
Z = aligned["coordinates"]

pyplot.draw()
pyplot.scatter(X[:,0], X[:,1])
pyplot.scatter(Z[:,0], Z[:,1])
pyplot.show()

# %% SciPy Procrustes 
from scipy.spatial import procrustes as pro 
A, B, d = pro(X, Y)
pyplot.draw()
pyplot.scatter(A[:,0], A[:,1])
pyplot.scatter(B[:,0], B[:,1])
pyplot.show()

# %% SciPy Procrustes rotation 
from scipy.linalg import orthogonal_procrustes as orthog_pro
R, s = orthog_pro(X, Y)

# %% Test scaled Procrustes rotation 
R, s = orthog_pro(X, Y*0.50)

# %% Draw the cover
pyplot.draw()
fig, ax = pyplot.subplots()
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[L["indices"],0], X[L["indices"],1], s=10)
for lm in L['indices']:
	c = pyplot.Circle((X[lm,0], X[lm,1]), r, color='orange', alpha = 0.20)
	ax.add_patch(c)
pyplot.show()

# %% Partition of unity 
from tallem.isomap import partition_of_unity
P = partition_of_unity(X, centers = X[L['indices'],:], radius = r)

# %% Phi Map 
# k == index of cover set, i = index of point 
from tallem.procrustes import ord_procrustes
from itertools import combinations
from tallem.distance import dist
from tallem.mds import classical_MDS

X = np.random.uniform(size=(15,2))
L = landmarks(X, k = 5)
r = np.min(L['radii'])

## Construct cover membership dictionary by landmark index
cover_membership = { i : [] for i in range(len(L['indices'])) }
for i in range(P.shape[0]):
	membership = np.where(P[i,:] > 0.0)[0]
	for c_idx in membership:
		cover_membership[c_idx].append(i)

## Create the local euclidean models 
# from sklearn.manifold import MDS
# metric_mds = MDS(n_components = 2)
# local_coords = [metric_mds.fit_transform(X[v,:]) for k, v in cover_membership.items()]
local_coords = [classical_MDS(dist(X[v,:], as_matrix=True)) for k, v in cover_membership.items()]

## Create the Phi map for each pair of cover sets
J = len(cover_membership.keys())

Omega = { "{},{}".format(j,k) : [] for j,k in combinations(range(J),2) }
for j,k in combinations(range(J),2):
	key = "{},{}".format(j,k)
	j_idx = cover_membership[j]
	k_idx = cover_membership[k]
	jk_idx = np.intersect1d(j_idx, k_idx)
	if len(jk_idx) > 1:
		rel_j_idx = np.array([cover_membership[j].index(elem) for elem in jk_idx])
		rel_k_idx = np.array([cover_membership[k].index(elem) for elem in jk_idx])
		j_coords = local_coords[j][rel_j_idx,:]
		k_coords = local_coords[k][rel_k_idx,:]
		Omega[key] = ord_procrustes(j_coords, k_coords, transform=False)
	else:
		Omega[key] = None


# %% Trying out joshes stuff 
import numpy as np
from tallem.landmark import landmarks
from tallem.sc import simp_comp, simp_search, delta0, delta0A, delta0D, landmark_cover, eucl_dist

## Get random data + landmarks 
X = np.random.uniform(size=(15,2))
L = landmarks(X, k = 5)
r = np.min(L['radii'])
cover = landmark_cover(X, L['indices'], r)

## 0-dimensional boundary matrix
S = [
	[0,1,2,3,4,5], 
	[[0,1], [2,3], [4,5]], 
	[1,2,3], [2,3,4]
]
delta0(S)

delta0A(S, w=dist(X[0:7,:], as_matrix=True))

# %%
import autograd.numpy as np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

# Need Stiefel(n=d*J,p=D) as Stiefel(n,p) := space of (n x p) orthonormal matrices
manifold = Stiefel(5*1000, 5)

def cost(X): 
	return np.sum(X)

problem = Problem(manifold=manifold, cost=cost)
solver = SteepestDescent()
Xopt = solver.solve(problem)


print(Xopt)





# %% TALLEM
