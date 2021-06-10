# %% Setup
%%time

## Assume as input: point cloud X and embedded coordinates F 
import numpy as np

## Map f: X -> B onto topological space 
f = F[:,1]
n = F.shape[0]

## Form basic partition of unity
## Note: nonzero entries describ the cover entirely 
from tallem.distance import dist
from tallem.isomap import partition_of_unity
poles = np.linspace(0, 2*np.pi, 10)
eps = 0.75*np.min(np.diff(poles))
f_mat = np.reshape(f, (len(f), 1))
p_mat = np.reshape(poles, (len(poles), 1))
def angular_dist(a,b):
	ddist = dist(a,b)
	odist = np.abs(ddist - 2*np.pi)
	return(np.minimum(ddist, odist))
P = partition_of_unity(f_mat, p_mat, eps, d=angular_dist)

## Identify the cover set for each point
point_cover_map = [np.ravel(P[i,:].nonzero())for i in range(n)]

## Aggregate points into cover->point map
nc = P.shape[1] # number of cover sets
cover_point_map = { i : np.ravel(P[:,i].nonzero()) for i in range(nc) }

## Use MDS coordinates to create the locla euclidean models
from tallem.mds import classical_MDS
Fj = { i : [] for i in range(nc) }
for i in range(nc):
	dx = dist(mobius_sample[cover_point_map[i],:], as_matrix=True) 
	Fj[i] = classical_MDS(dx, k=2)

## Extract intersection maps 
def intersection_map(pair):
	i, j = pair
	ij_ind = np.intersect1d(cover_point_map[i], cover_point_map[j])
	if len(ij_ind) > 0:
		i_idx = np.ravel([np.where(np.array(cover_point_map[i]) == element) for element in ij_ind])
		j_idx = np.ravel([np.where(np.array(cover_point_map[j]) == element) for element in ij_ind])
		return({ "indices" : ij_ind, "model1" : Fj[i][i_idx,:], "model2": Fj[j][j_idx,:] })
	else: 
		v = np.empty((0,), np.float64)
		return({ "indices": np.asarray(v, dtype=np.int32), "model1": v, "model2": v })

## Solve Procrustes problem for each non-empty intersection
from itertools import combinations
from tallem.procrustes import ord_procrustes
Omega_map = {}
edges = []
for (i,j) in combinations(range(nc), 2):
	F_models = intersection_map((i,j))
	if len(F_models["indices"]) > 1:
		Omega_map[(i,j)] = ord_procrustes(F_models["model1"], F_models["model2"], transform=False)


## Phi map: supplying i returns the phi map for the largest 
## value in the partition of unity; supplying j as well returns 
## the specific phi map for the jth cover element (possibly 0)
def phi(i, j = None):
	J, d = P.shape[1], F.shape[1]
	k = np.argmax(P[i,:]) if j is None else j
	out = np.zeros((d*J, d))
	def weighted_omega(j):
		nonlocal k
		w = np.sqrt(P[i,j])
		pair_exists = [pair in Omega_map.keys() for pair in [(k,j), (j,k)]]
		if w == 0.0 or not(np.any(pair_exists)):
			return(w*np.eye(d))
		return(w*Omega_map[(k,j)]['rotation'] if pair_exists[0] else w*Omega_map[(j,k)]['rotation'].T)
	return(np.vstack([weighted_omega(j) for j in range(J)]))

# %% Optimization to find the best A matrix 
%%time

## First get the initial guess A_0
## Based on joshes frame_init.m code 
## TODO: repalce with scipy.sparse.linalg.eigs(k=D)
## Given target dimension 'D', phi map 'phi' and the number of points mapped by phi 'n', produces the
## initial projection frame A which maximizes sum of Frobenius norm || A.T @ phi(x) ||_F for all x \in X
def initial_frame(D, phi, n):
	Fb = np.vstack([phi(j).T for j in range(n)])
	Phi_N = Fb.T @ Fb
	Eval, Evec = np.linalg.eigh(Phi_N)
	return(Evec[:,np.argsort(-Eval)[:D]])

import autograd.scipy.linalg as auto_scipy 
import autograd.numpy as auto_np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

# Need Stiefel(n=d*J,p=D) as Stiefel(n,p) := space of (n x p) orthonormal matrices
D = 3 # desired embedding dimension 
manifold = Stiefel(d*nc, D)

def huber_loss(epsilon: auto_np.float32 = 1.0, update_eps = lambda eps: eps):
	def cost_function(A):
		nonlocal epsilon 
		nuclear_norm = 0.0
		for j in range(n):
			M = auto_np.array(A.T @ phi(j), dtype='float')
			svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False)
			nuclear_norm += auto_np.sum([t if t >= epsilon else (t**2)/epsilon for t in auto_np.abs(svals)])
		epsilon = update_eps(epsilon)
		return(-nuclear_norm)
	return(cost_function)

problem = Problem(manifold=manifold, cost=huber_loss(1.0))
solver = SteepestDescent()
A0 = initial_frame(D, phi, X.shape[0])
Xopt = solver.solve(problem=problem, x=A0)

# %% Get optimal translation vectors 
%%time
from tallem.sc import delta0D
S = (np.fromiter(range(nc), dtype=int), list(Omega_map.keys()))
Omega = { k: v['rotation'] for k,v in Omega_map.items() }
Trans = { k: v['translation'] for k,v in Omega_map.items() }

## Evaluates pseudo-inverse on the coboundary matrix    
deltaX = np.linalg.pinv(delta0D(S, Omega))
shiftE = np.zeros(d*len(S[1]))
for (index,(i1,i2)) in enumerate(S[1]):
		shiftE[index*d:(index+1)*d] = Trans[(i1,i2)]
shiftV = np.matmul(deltaX,shiftE)

## Offsets contain the translation vectors, keyed by edge indices in the nerve complex 
offsets = {}
for index in range(J):
	offsets[index] = shiftV[index*d:(index+1)*d]

# %% Construct the global assembly function 
%%time
A_opt = Xopt
A_cov = A_opt @ A_opt.T
assembly = np.zeros((n, D), dtype=np.float64)
for i in range(n):
	idx = np.ravel(P[i,:].nonzero())
	coords = np.zeros((1,D), dtype=np.float64)
	for j in idx:
		w = P[i,j]
		if i in cover_point_map[j] and w > 0.0:
			u, s, vt = np.linalg.svd(A_cov @ phi(i, j), full_matrices=False)
			i_idx = np.ravel(np.where(cover_point_map[j] == i))
			f_x = Fj[j][i_idx,:] + offsets[j]
			coords += w * (A_opt.T @ (u @ vt) @ f_x.T).T
	assembly[i,:] = coords

## View the assembly 
ax = pyplot.axes(projection='3d')
ax.scatter3D(assembly[:,0], assembly[:,1], assembly[:,2], c=F[:,0],s=1.50)

ax = pyplot.axes(projection='3d')
ax.scatter3D(assembly[:,0], assembly[:,1], assembly[:,2], c=F[:,1],s=1.50)



# %% Benchmarking various SVD solutions to computing singular values
# from scipy.sparse import random as sparse_random
# from sklearn.utils.extmath import randomized_svd
# from scipy.linalg import svdvals
# from jax.numpy.linalg import svd as jax_svd
# X = sparse_random(100, 100, density=0.01, format='csr')
# X_np = X.toarray()

# run_sec_vals = timeit.repeat(lambda: svdvals(X_np), number=50)
# run_sec_full = timeit.repeat(lambda: np.linalg.svd(X_np, full_matrices=False), number=50)
# run_sec_random = timeit.repeat(lambda: randomized_svd(X, n_components=2), number=50)
# run_sec_jax = timeit.repeat(lambda: jax_svd(X_np, full_matrices=False, compute_uv=False), number=50)
# run_sec_norm = timeit.repeat(lambda: np.linalg.norm(X_np, 'nuc'), number=50) 

# res1 = "(100 x 100) Values: {0:.3f} ms, Full: {1:.3f} ms, Randomized: {2:.3f} ms, JAX: {3:.3f}, norm: {4:.3f}".format(
# 	np.mean(run_sec_vals*1000), 
# 	np.mean(run_sec_full*1000),
# 	np.mean(run_sec_random*1000),
# 	np.mean(run_sec_jax*1000), 
# 	np.mean(run_sec_norm*1000)
# )
# print(res1)

# X = sparse_random(10000, 10, density=0.10, format='csc')
# X_np = X.toarray()

# run_sec_vals = timeit.repeat(lambda: svdvals(X_np), number=50)
# run_sec_full = timeit.repeat(lambda: np.linalg.svd(X_np, full_matrices=False), number=50)
# run_sec_random = timeit.repeat(lambda: randomized_svd(X, n_components=2), number=50)
# run_sec_jax = timeit.repeat(lambda: jax_svd(X_np, full_matrices=False, compute_uv=False), number=50)
# run_sec_norm = timeit.repeat(lambda: np.linalg.norm(X_np, 'nuc'), number=50) 

# res2 = "(10000 x 10) Values: {0:.3f} ms, Full: {1:.3f} ms, Randomized: {2:.3f} ms, JAX: {3:.3f}, norm: {4:.3f}".format(
# 	np.mean(run_sec_vals*1000), 
# 	np.mean(run_sec_full*1000),
# 	np.mean(run_sec_random*1000),
# 	np.mean(run_sec_jax*1000),
# 	np.mean(run_sec_norm*1000)
# )
# print(res2)

# randomized_svd(X, n_components=2)


# def filter_coords(pair):
# 	ij_idx = np.intersect1d(cover_point_map[pair[0]], cover_point_map[pair[1]])
# 	return(F[ij_idx,:] if len(ij_idx) > 1 else None)

# v = { (i,j) : filter_coords((i,j)) for (i,j) in combinations(range(nc), 2) }