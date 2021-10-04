# %% Dimensionality reduction imports 
import os
import numpy as np
import numpy.typing as npt
import concurrent.futures
from typing import Optional, Dict, List, Union
from .distance import dist, is_distance_matrix
from .utility import as_np_array
from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eig as dense_eig
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


# %%  Dimensionality reduction definitions
def pca(x: npt.ArrayLike, d: int = 2, center: bool = True) -> npt.ArrayLike:
	''' PCA embedding '''
	assert not(is_distance_matrix(x)), "Input should be a point cloud, not a distance matrix."
	if center: x -= x.mean(axis = 0)
	ew, ev = np.linalg.eigh(np.cov(x, rowvar=False))
	idx = np.argsort(ew)[::-1] # descending order to pick the largest components first 
	return(np.dot(x, ev[:,idx[range(d)]]))

def sammon(data, d: int = 2, max_iterations: int = 250, max_halves: int = 10):
	"""
	This implementation is adapted from a (1) GSOC 2016 project by Daniel McNeela (1) 
	which itself is adapted from the (2) Matlab implementation by Dr. Gavin C. Cawley.
	
	Sources can be found here: 
	1. https://github.com/mcneela/Retina
	2. https://people.sc.fsu.edu/~jburkardt/m_src/profile/sammon_test.m
	"""
	TolFun = 1 * 10 ** (-9)
	
	D = dist(data, as_matrix = True)
	N = data.shape[0]
	scale = np.sum(D.flatten('F'))
	D = D + np.identity(N)
	D_inv = np.linalg.inv(D)
	
	y = np.random.randn(N, d)
	one = np.ones((N, d))
	d = dist(y, as_matrix = True) + np.identity(N)
	d_inv = np.linalg.inv(d)
	delta = D - d
	E = np.sum(np.sum(np.power(delta, 2) * D_inv))

	for i in range(max_iterations):
		delta = d_inv - D_inv
		deltaone = np.dot(delta, one)
		g = np.dot(delta, y) - y * deltaone
		dinv3 = np.power(d_inv, 3)
		y2 = np.power(y, 2)
		H = np.dot(dinv3, y2) - deltaone - 2 * np.multiply(y, np.dot(dinv3, y)) + np.multiply(y2, np.dot(dinv3, one))
		s = np.divide(-np.transpose(g.flatten('F')), np.transpose(np.abs(H.flatten('F'))))
		y_old = y

	for j in range(max_halves):
		[rows, columns] = y.shape
		y = y_old.flatten('F') + s
		y = y.reshape(rows, columns)
		d = dist(y, as_matrix = True) + np.identity(N)
		d_inv = np.linalg.inv(d)
		delta = D - d
		E_new = np.sum(np.sum(np.power(delta, 2) * D_inv))
		if E_new < E:
			break
		else:
			s = 0.5 * s

	E = E_new
	E = E * scale
	return (y, E)

## Classical MDS 
def cmds(a: npt.ArrayLike, d: int = 2, coords: bool = True):
	''' Computes classical MDS (cmds) w/ a being a distance matrix '''
	a = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	n, m = a.shape
	if n <= 1: return(np.repeat(0.0, m))
	C = np.eye(n) - (1.0/n)*np.ones(shape=(n,n))
	B = (-1/2)*(C @ a @ C)
	if d >= (n-1):
		E = dense_eig(B)
		if d == (n - 1):
			max_idx = np.argsort(-E[0])[:(n-1)]
			E = ([E[0][max_idx], E[1][:,max_idx]]) 
	else: 
		E = truncated_eig(B, k = d)
	## Eigenvalues should be real since matrix was symmetric 
	E = (np.real(E[0]), np.real(E[1]))
	return(E[0] * E[1] if coords else E)

def neighborhood_graph(a: npt.ArrayLike, k: Optional[int] = 15, radius: Optional[float] = None, **kwargs):
	''' 
	Computes the neighborhood graph of a point cloud or distance matrix 'a'. 
	Returns a sparse weighted adjacency matrix where positive entries indicate the distance between points in 'a'. 
	'''
	return(neighborhood_list(a, a, k, radius, **kwargs))
	# if radius is None and k is None: raise RuntimeError("Either radius or k must be supplied")
	# a = as_np_array(a)
	# n = a.shape[0]

	# ## If 'a' is a point cloud, form a KD tree to extract the neighbors
	# if not(is_distance_matrix(a)):
	# 	tree = KDTree(data=a, **kwargs)
	# 	if radius is not None:
	# 		pairs = tree.query_pairs(r=radius*2.0)
	# 		r, c = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
	# 		d = dist(a[r,:], a[c,:], pairwise = True)
	# 	else:
	# 		knn = tree.query(a, k=k+1)
	# 		r, c, d = np.repeat(range(n), repeats=k), knn[1][:,1:].flatten(), knn[0][:,1:].flatten()
	# else: 
	# 	if radius is not None: 
	# 		r, c = np.where(a <= (radius*2.0))
	# 		valid = (r != c) & (r < c)
	# 		r, c = r[valid], c[valid]
	# 		d = a[r,c]
	# 	else: 
	# 		knn = np.apply_along_axis(lambda a_row: np.argsort(a_row)[0:(k+1)],axis=1,arr=a)
	# 		r, c = np.repeat(range(n), repeats=k), np.ravel(knn[:,1:])
	# 		d = a[r,c]

	# ## Form the neighborhood graph 
	# D = csc_matrix((d, (r, c)), dtype=np.float32, shape=(n,n))
	return(D)


def neighborhood_list(centers: npt.ArrayLike, a: npt.ArrayLike, k: Optional[int] = 15, radius: Optional[float] = None, metric = "euclidean", **kwargs):
	''' 
	Computes the neighborhood adjacency list of a point cloud 'centers' using points in 'a'. 
	If 'a' is a (n x d) matrix and 'centers' is a (m x d) matrix, this function computes a sparse (n x m) matrix 
	where the non-zero entries I at each column j are the 'metric' distances from point j in 'b' to the center points.
	'''
	assert k >= 1, "k must be an integer, if supplied" 
	a, b = as_np_array(a), as_np_array(centers) # b is the centers
	n, m = a.shape[0], b.shape[0]
	minkowski_metrics = ["cityblock", "euclidean", "chebychev"]
	if metric in minkowski_metrics:
		p = [1, 2, float("inf")][minkowski_metrics.index(metric)]
		if radius is not None:
			tree = KDTree(data=a, **kwargs)
			neighbors = tree.query_ball_point(b, r=radius, p=p)
			r = np.array(np.hstack(neighbors), dtype=np.int32)
			c = np.repeat(range(m), repeats=[len(idx) for idx in neighbors])
			d = dist(a[r,:], b[c,:], pairwise=True, metric=metric)
			# for i, nn_idx in enumerate(tree.query_ball_point(a, r=radius, p=p)):
			# 	G[i,nn_idx] = dist(a[[i],:], b[nn_idx,:], metric=metric)
		else:
			tree = KDTree(data=b, **kwargs)
			knn = tree.query(a, k=k)
			r, c, d = np.repeat(range(a.shape[0]), repeats=k), knn[1].flatten(), knn[0].flatten()
	else: 
		D = dist(a, b, metric=metric)
		if radius is not None: 
			I = np.argwhere(D <= radius)
			r, c = I[:,0], I[:,1]
			d = D[r,c]
		else: 
			knn = np.apply_along_axis(lambda a_row: np.argsort(a_row)[0:k],axis=1,arr=D)
			r, c = np.repeat(range(n), repeats=k), np.ravel(knn)
			d = D[r,c]
	G = csc_matrix((d, (r, c)), dtype=np.float32, shape=(max(max(r), n), max(max(c), m)))
	return(G)

def floyd_warshall(a: npt.ArrayLike):
	'''floyd_warshall(adjacency_matrix) -> shortest_path_distance_matrix
	Input
			An NxN NumPy array describing the directed distances between N nodes.
			adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)
			Notes:
			* If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
			* The diagonal of adjacency_matrix should be zero.
			Based on https://gist.github.com/mosco/11178777
	Output
			An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
	'''
	a = as_np_array(a)
	n = a.shape[0]
	a[a == 0.0] = np.inf
	np.fill_diagonal(a, 0.0) # Ensure diagonal is 0!
	for k in range(n): a = np.minimum(a, a[np.newaxis,k,:] + a[:,k,np.newaxis])
	return(a)

def connected_radius(a: npt.ArrayLike) -> float:
	''' Returns the smallest 'r' such that the union of balls of radius 'r' space is connected'''
	a = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	return(np.max(minimum_spanning_tree(csr_matrix(a))))

def enclosing_radius(a: npt.ArrayLike) -> float:
	''' Returns the smallest 'r' such that the Rips complex on the union of balls of radius 'r' is contractible to a point. '''
	a = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	return(np.min(np.amax(a, axis = 0)))
 
def geodesic_dist(a: npt.ArrayLike):
	d = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	return(floyd_warshall(d))

from sklearn.manifold import Isomap
def isomap(a: npt.ArrayLike, d: int = 2, **kwargs) -> npt.ArrayLike:
	''' Returns the isomap embedding of a given point cloud or distance matrix. '''
	if "k" in kwargs.keys():
		metric = "euclidean" if not("metric" in kwargs.keys()) else kwargs["metric"]
		E = Isomap(n_neighbors=kwargs["k"], n_components=d, metric=metric)
		return(E.fit_transform(a))
	else: 
		G = neighborhood_graph(np.asanyarray(a), **kwargs)
		assert connected_components(G, directed = False)[0] == 1, "Error: graph not connected. Can only run isomap on a fully connected neighborhood graph."
		return(cmds(geodesic_dist(G.todense()), d))


# TODO: remove sklearn eventually
from sklearn.manifold import MDS
def mmds(a: npt.ArrayLike, d: int = 2, **kwargs):
	''' Thin wrapper around sklearn's metric MDS '''
	emb = MDS(n_components=d, metric=True,  dissimilarity='precomputed', random_state=0, **kwargs) if is_distance_matrix(a) else MDS(n_components=d, metric=True, random_state=0, **kwargs) 
	return(emb.fit_transform(a))

def nmds(a: npt.ArrayLike, d: int = 2, **kwargs):
	''' Thin wrapper around sklearn's non-metric MDS '''
	embedding = MDS(n_components=d, metric=False, random_state=0, **kwargs)
	return(embedding.fit_transform(a))


#from numba import njit, prange
# @njit(parallel=True)
# def fit_local_models(f, X, cover):
# 	index_set = list(cover.keys())
# 	subsets = list(cover.values())
# 	result = {}
# 	for j in prange(len(cover)):
# 		index, subset = index_set[j], subsets[j]
# 		result[index] = f(X[np.array(subset),:])
# 	return(result)

def fit_local_models(f, X, cover, n_cores=1): #os.cpu_count()
	if n_cores == 1:
		models = { index : f(X[np.array(subset),:]) for index, subset in cover.items() }
	else:
		models = {}
		do_euclidean_model = lambda ce: (ce[0], f(X[np.array(ce[1]),:])) 
		with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
			future = executor.map(do_euclidean_model, cover.items())
			for index, model in future:
				models[index] = model
	return(models)
