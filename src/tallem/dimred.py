# %% Dimensionality reduction imports 
import os
import numbers
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import *
from .distance import *
from .utility import *
from .samplers import landmarks
from .extensions import mds_cython

from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eigh, eig as dense_eig
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

# %%  Dimensionality reduction definitions
def pca(x: npt.ArrayLike, d: int = 2, center: bool = False, coords: bool = True) -> npt.ArrayLike:
	''' PCA embedding '''
	if is_pairwise_distances(x) or is_distance_matrix(x):
		return(cmds(x, d))
	assert is_point_cloud(x), "Input should be a point cloud, not a distance matrix."
	if center: x -= x.mean(axis = 0)
	evals, evecs = np.linalg.eigh(np.cov(x, rowvar=False))
	idx = np.argsort(evals)[::-1] # descending order to pick the largest components first 
	if coords:
		return(np.dot(x, evecs[:,idx[range(d)]]))
	else: 
		return(np.flip(evals)[range(d)], np.fliplr(evecs)[:,range(d)])

## Classical MDS 
def cmds(a: npt.ArrayLike, d: int = 2, coords: bool = True, method="fortran"):
	''' Computes classical MDS (cmds) '''
	if is_pairwise_distances(a):
		D = as_dist_matrix(a)
	elif not(is_distance_matrix(a)) and is_point_cloud(a):
		D = dist(a, as_matrix=True, metric="euclidean")**2
	else:
		D = a
	assert(is_distance_matrix(D))
	n = D.shape[0]
	# mds_cython.double_center(D, n) # double-centers D inplace
	if method == "scipy":
		H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
		evals, evecs = eigh(-0.5 * H @ D @ H, subset_by_index=(n-d, n-1))
	elif method == "numpy": 
		H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
		evals, evecs = np.linalg.eigh(-0.5 * H @ D @ H)
		evals, evecs = evals[(n-d):n], evecs[:,(n-d):n]
	else:
		D_center = D.mean(axis=0)
		D = -0.50 * (D  - D_center - D_center.reshape((n,1)) + D_center.mean())
		evals, evecs = mds_cython.cython_dsyevr(D, n-d+1, n, 1e-8, False)

	# Compute the coordinates using positive-eigenvalued components only     
	if coords:               
		w = np.flip(np.maximum(evals, np.repeat(0.0, d)))
		Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
		return(Y)
	else: 
		w = np.where(evals > 0)[0]
		ni = np.setdiff1d(np.arange(d), w)
		evecs[:,ni] = 1.0
		evals[ni] = 0.0
		return(evals, evecs)


def landmark_mds(X: ArrayLike, d: int = 2, L: Union[ArrayLike, int, str] = "default", normalize=False, ratio=1.0, prob=1.0):
	''' 
	Landmark Multi-Dimensional Scaling 
	
	Parameters: 
		X := point cloud matrix, distance matrix, set of pairwise distances.
		d := target dimension for the coordinatization
		L := either an integer specifying the number of landmarks to use, or indices of 'X' designating which landmarks to use
		normalize := whether to re-orient the embedding using PCA to reflect the distribution of 'X' rather than distribution of landmarks. Defaults to false.   
		ratio := aspect ratio between the smallest/largest dimensions of the bounding box containing 'X'. Defaults to 1. See details.
		prob := probability the embedding should match exactly with the results of MDS. See details. 

	Details: 
		This function uses landmark	points and trilateration to compute an approximation to the embedding obtained by classical 
		multidimensional scaling, using the technique described in [1].

		The parameter 'L' can be either an array of indices of the rows of 'X' indicating which landmarks to use, a single integer specifying the 
		number of landmarks to compute using maxmin, or "default" in which case the number of landmarks to use is calculated automatically. In the 
		latter case, 'ratio' and 'prob' are used to calculate the number of landmarks needed to recover the same embedding one would obtain using 
		classical MDS on the full (squared) euclidean distance matrix of 'X'. The bound is from [2], which sets the number of landmarks 'L' to: 

		L = floor(9*(ratio**2)*log(2*(d+1)/prob))

		Since this bound was anlyzed with respect to uniformly random samples, it tends to overestimate the number of landmarks needed compared to 
		using the maxmin approach, which is much more stable. In general, a good rule of thumb is choose L as some relatively small multiple of the 
		target dimension d, i.e. something like L = 15*d.

	References: 
		1. De Silva, Vin, and Joshua B. Tenenbaum. Sparse multidimensional scaling using landmark points. Vol. 120. technical report, Stanford University, 2004.
		2. Arias-Castro, Ery, Adel Javanmard, and Bruno Pelletier. "Perturbation bounds for procrustes, classical scaling, and trilateration, with applications to manifold learning." Journal of machine learning research 21 (2020): 15-1.

	'''
	if isinstance(L, str) and (L == "default"):
		L = int(9*(ratio**2)*np.log(2*(d+1)/prob))
		subset = landmarks(X, k=L)
	elif isinstance(L, numbers.Integral):
		subset = landmarks(X, k=L)
	else: 
		assert isinstance(L, np.ndarray)
		subset = L

	## Apply classical MDS to landmark points
	from itertools import combinations
	J = len(subset) 
	if is_pairwise_distances(X):
		D, n = as_dist_matrix(subset_dist(X, subset)), inverse_choose(len(X), 2)
		S = np.zeros(shape=(J,n))
		for j, index in enumerate(subset):
			for i in range(n):
				S[j,i] = 0.0 if i == index else X[rank_comb2(i,index,n)]
	elif is_distance_matrix(X):
		D, n = subset_dist(X, subset), X.shape[0]
		S = X[np.ix_(subset, range(n))]
	else:
		D, n = dist(X[subset,:], as_matrix=True, metric="euclidean")**2, X.shape[0]
		S = dist(X[subset,:], X, metric="euclidean")**2
	
	## At this point, D == distance matrix of landmarks points, S == (J x n) distances to landmarks
	evals, evecs = cmds(D, d=d, coords=False)

	## Interpolate the lower-dimension points using the landmarks
	mean_landmark = np.mean(D, axis = 1).reshape((D.shape[0],1))
	w = np.where(evals > 0)[0]
	L_pseudo = evecs/np.sqrt(evals[w])
	Y = np.zeros(shape=(n, d))
	Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 

	## Normalize using PCA, if requested
	if (normalize):
		m = Y.shape[0]
		Y_hat = Y.T @ (np.eye(m) - (1.0/m)*np.ones(shape=(m,m)))
		_, U = np.linalg.eigh(Y_hat @ Y_hat.T) # Note: Y * Y.T == (k x k) matrix
		Y = (U.T @ Y_hat).T
	return(Y)

def landmark_isomap(X: ArrayLike, d: int = 2, L: Union[ArrayLike, int, str] = "default", normalize=False, ratio=1.0, prob=1.0, **kwargs):
	
	## Compute the landmarks
	if isinstance(L, str) and (L == "default"):
		L = int(9*(ratio**2)*np.log(2*(d+1)/prob))
		subset = landmarks(X, k=L)
	elif isinstance(L, numbers.Integral):
		subset = landmarks(X, k=L)
	else: 
		assert isinstance(L, np.ndarray)
		subset = L

	## Compute the neighborhood graph
	if "radius" in kwargs.keys():
		G = rnn_graph(X, r=kwargs["radius"])
	elif "k" in kwargs.keys():
		G = knn_graph(X, k=kwargs["k"])
	else: 
		G = rnn_graph(X, **kwargs) # should pick up defaults if given

	## Compute the distances from every point to every landmark, and the landmark distance matrix
	S = csgraph.dijkstra(G, directed=False, indices=subset, return_predecessors=False)
	D = S[:,subset]
	
	## At this point, D == distance matrix of landmarks points, S == (J x n) distances to landmarks
	evals, evecs = cmds(D, d=d, coords=False)

	## Interpolate the lower-dimension points using the landmarks
	mean_landmark = np.mean(D, axis = 1).reshape((D.shape[0],1))
	w = np.where(evals > 0)[0]
	L_pseudo = evecs/np.sqrt(evals[w])
	Y = np.zeros(shape=(n, d))
	Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 

	## Normalize using PCA, if requested
	if (normalize):
		m = Y.shape[0]
		Y_hat = Y.T @ (np.eye(m) - (1.0/m)*np.ones(shape=(m,m)))
		_, U = np.linalg.eigh(Y_hat @ Y_hat.T) # Note: Y * Y.T == (k x k) matrix
		Y = (U.T @ Y_hat).T
	return(Y)


def neighborhood_graph(a: npt.ArrayLike, k: Optional[int] = 15, radius: Optional[float] = None, **kwargs):
	''' 
	Computes the neighborhood graph of a point cloud or distance matrix 'a'. 
	Returns a sparse weighted adjacency matrix where positive entries indicate the distance between points in 'a'. 
	'''
	return(neighborhood_list(a, a, k, radius, **kwargs))

## Note: neighborhood list doesn't work with distance matrices!
def neighborhood_list(centers: npt.ArrayLike, a: npt.ArrayLike, k: Optional[int] = 15, radius: Optional[float] = None, metric = "euclidean", **kwargs):
	''' 
	Computes the neighborhood adjacency list of a point cloud 'centers' using points in 'a'. 
	If 'a' is a (n x d) matrix and 'centers' is a (m x d) matrix, this function computes a sparse (n x m) matrix 
	where the non-zero entries I at each column j are the 'metric' distances from point j in 'b' to the center points.
	'''
	assert k >= 1, "k must be an integer, if supplied" 
	minkowski_metrics = ["cityblock", "euclidean", "chebychev"]
	if is_point_cloud(centers) and is_point_cloud(a) and (metric in minkowski_metrics):
		n,m = a.shape[0], centers.shape[0]
		p = [1, 2, float("inf")][minkowski_metrics.index(metric)]
		if radius is not None:
			tree = KDTree(data=a, **kwargs)
			neighbors = tree.query_ball_point(centers, r=radius, p=p)
			r = np.array(np.hstack(neighbors), dtype=np.int32)
			c = np.repeat(range(m), repeats=[len(idx) for idx in neighbors])
			d = dist(a[r,:], centers[c,:], pairwise=True, metric=metric)
			# for i, nn_idx in enumerate(tree.query_ball_point(a, r=radius, p=p)):
			# 	G[i,nn_idx] = dist(a[[i],:], b[nn_idx,:], metric=metric)
		else:
			tree = KDTree(data=a, **kwargs)
			knn = tree.query(centers, k=k)
			r, c, d = np.repeat(range(a.shape[0]), repeats=k), knn[1].flatten(), knn[0].flatten()
	elif is_point_cloud(centers) and is_point_cloud(a): 
		D = dist(a, centers, metric=metric)
		if radius is not None: 
			I = np.argwhere(D <= radius)
			r, c = I[:,0], I[:,1]
			d = D[r,c]
		else: 
			knn = np.apply_along_axis(lambda a_row: np.argsort(a_row)[0:k],axis=1,arr=D)
			r, c = np.repeat(range(n), repeats=k), np.ravel(knn)
			d = D[r,c]
	elif is_dist_like(a):
		assert is_index_like(centers), "If distances are given, 'centers' must be an index vector"
		n = inverse_choose(len(a), 2) if is_pairwise_distances(a) else a.shape[0]
		D = subset_dist(a, (centers, range(n)))
		if radius is not None: 
			I = np.argwhere(D <= radius)
			r, c = I[:,0], I[:,1]
			d = D[r,c]
		else: 
			knn = np.apply_along_axis(lambda a_row: np.argsort(a_row)[0:k],axis=1,arr=D)
			r, c = np.repeat(range(n), repeats=k), np.ravel(knn)
			d = D[r,c]
	else: 
		raise ValueError("Invalid input. Only accepts dist-like objects and point clouds.")
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
	assert is_distance_matrix(a)
	return(np.max(minimum_spanning_tree(a)))

def enclosing_radius(a: npt.ArrayLike) -> float:
	''' Returns the smallest 'r' such that the Rips complex on the union of balls of radius 'r' is contractible to a point. '''
	assert is_distance_matrix(a)
	return(np.min(np.amax(a, axis = 0)))
 
def geodesic_dist(a: npt.ArrayLike):
	d = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	return(floyd_warshall(d))

def rnn_graph(a: npt.ArrayLike, r: Optional[float] = None, p = 0.15):
	D = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	if r is None:
		assert isinstance(p, float) and p >= 0.0
		cr, er = connected_radius(D), enclosing_radius(D)
		r = cr + p*(er-cr)
	return(neighborhood_graph(np.asanyarray(a), radius=r))

def knn_graph(a: npt.ArrayLike, k: Optional[int] = 15):
	D = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	if k is None: 
		k = 15
	return(neighborhood_graph(np.asanyarray(a), k = k))

def isomap(a: npt.ArrayLike, d: int = 2, **kwargs) -> npt.ArrayLike:
	''' 
	Returns the isomap embedding of a given point cloud. 
	
	Parameters: 
		a := point cloud matrix. 
		d := (optional) target dimension of the embedding (defaults to 2).
		k := (optional) number of nearest neighbors to connect to form the neighborhood graph
		r := (optional) the radius of each ball centered around each point in 'a' to form the neighborhood graph
		p := (optional) proportion between the connecting radius and the enclosing radius to calculate 'r' (between [0,1])

	Exactly one of the parameters (k,r,p) should be chosen to determine the connectivity of the graph. The default, if
	none is chosen, is to pick p = 0.15. 
	'''
	if "radius" in kwargs.keys():
		G = neighborhood_graph(np.asanyarray(a), **kwargs)
		assert connected_components(G, directed = False)[0] == 1, "Error: graph not connected. Can only run isomap on a fully connected neighborhood graph."
		return(cmds(geodesic_dist(G.A), d))
	elif "k" in kwargs.keys():
		ask_package_install("sklearn")
		from sklearn.manifold import Isomap
		metric = "euclidean" if not("metric" in kwargs.keys()) else kwargs["metric"]
		E = Isomap(n_neighbors=kwargs["k"], n_components=d, metric=metric)
		return(E.fit_transform(a))
	else: 
		G = rnn_graph(a, **kwargs)
		assert connected_components(G, directed = False)[0] == 1, "Error: graph not connected. Can only run isomap on a fully connected neighborhood graph."
		return(cmds(geodesic_dist(G.A), d))

def mmds(a: npt.ArrayLike, d: int = 2, **kwargs):
	''' Thin wrapper around sklearn's metric MDS '''
	ask_package_install("sklearn")
	from sklearn.manifold import MDS
	emb = MDS(n_components=d, metric=True,  dissimilarity='precomputed', random_state=0, **kwargs) if is_distance_matrix(a) else MDS(n_components=d, metric=True, random_state=0, **kwargs) 
	return(emb.fit_transform(a))

def nmds(a: npt.ArrayLike, d: int = 2, **kwargs):
	''' Thin wrapper around sklearn's non-metric MDS '''
	ask_package_install("sklearn")
	from sklearn.manifold import MDS
	embedding = MDS(n_components=d, metric=False, random_state=0, **kwargs)
	return(embedding.fit_transform(a))

def fit_local_models(f, X, cover, n_cores=1): #os.cpu_count()
	models = { index : f(X[np.array(subset),:]) for index, subset in cover.items() }
	# if n_cores == 1:
	# 	
	# else:
	# 	models = {}
	# 	do_euclidean_model = lambda ce: (ce[0], f(X[np.array(ce[1]),:])) 
	# 	with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
	# 		future = executor.map(do_euclidean_model, cover.items())
	# 		for index, model in future:
	# 			models[index] = model
	return(models)
