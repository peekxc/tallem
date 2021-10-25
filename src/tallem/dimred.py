# %% Dimensionality reduction imports 
import os
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import *
from .distance import *
from .utility import *
from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eigh, eig as dense_eig
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

## Classical MDS 
def cmds(a: npt.ArrayLike, d: int = 2, coords: bool = True, method="scipy"):
	''' Computes classical MDS (cmds) '''
	if is_pairwise_distances(a):
		D = as_dist_matrix(a)
	elif not(is_distance_matrix(a)) and is_point_cloud(a):
		D = dist(a, as_matrix=True, metric="euclidean")**2
	else:
		D = a
	assert(is_distance_matrix(D))
	n = D.shape[0]
	H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
	B = -0.5 * H @ D @ H
	if method == "scipy":
		evals, evecs = eigh(B, subset_by_index=(n-d, n-1))
	else: 
		evals, evecs = np.linalg.eigh(B)
		evals, evecs = np.flip(evals), np.fliplr(evecs)
		evals, evecs = evals[range(d)], evecs[:,range(d)]
	
	# Compute the coordinates using positive-eigenvalued components only     
	if coords:               
		w = np.where(evals > 0)[0]
		Y = np.zeros(shape=(n, d))
		Y[:,w] = evecs[:,w] @ np.diag(np.sqrt(evals[w]))
		return(Y)
	else: 
		w = np.where(evals > 0)[0]
		ni = np.setdiff1d(np.arange(d), w)
		evecs[:,ni] = 1.0
		evals[ni] = 0.0
		return(evals, evecs)

def landmark_mds(X: ArrayLike, subset: ArrayLike, d: int = 2, normalize=False):
	''' 
	Landmark Multi-dimensional Scaling 
	
	Parameters: 
		X := distance matrix, set of pairwise distances, or a point cloud
		subset := indices of landmarks to use
		d := target dimension for the coordinatization
		
	'''
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

def rnn_graph(a: npt.ArrayLike, r: Optional[float] = None, p = 0.15):
	# from sklearn.neighbors import radius_neighbors_graph
	D = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	if r is None:
		cr, er = connected_radius(D), enclosing_radius(D)
		r = cr + p*(er-cr)
	return(neighborhood_graph(np.asanyarray(a), radius=r))
	# radius_neighbors_graph(x, radius = )

def knn_graph(a: npt.ArrayLike, k: Optional[int] = 15):
	D = dist(a, as_matrix=True) if not(is_distance_matrix(a)) else np.asanyarray(a)
	if k is None: 
		k = 15
	return(neighborhood_graph(np.asanyarray(a), k = k))

def isomap(a: npt.ArrayLike, d: int = 2, **kwargs) -> npt.ArrayLike:
	''' 
	Returns the isomap embedding of a given point cloud or distance matrix. 
	
	Parameters: 
		a := point cloud matrix. 
		d := target dimension of the embedding (default to 2).
		k := number of nearest neighbors to connect to form the neighborhood graph
		r := the radius of each ball centered around each point in 'a' to form the neighborhood graph
		p := proportion between the connecting radius and the enclosing radius to use a ball radius (between [0,1])

	The parameters
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
