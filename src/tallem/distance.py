# %% Distance imports
import numbers
import numpy as np
import numpy.typing as npt 
from numpy.typing import ArrayLike
from typing import *
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.sparse import issparse
from .utility import inverse_choose, rank_comb2, unrank_comb2

# %% Distance definitions
def is_distance_matrix(x: ArrayLike) -> bool:
	''' Checks whether 'x' is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0. '''
	x = np.array(x, copy=False)
	is_square = x.ndim == 2	and (x.shape[0] == x.shape[1])
	return(False if not(is_square) else np.all(np.diag(x) == 0))

def is_pairwise_distances(x: ArrayLike) -> bool:
	''' Checks whether 'x' is a 1-d array of pairwise distances '''
	x = np.array(x, copy=False) # don't use asanyarray here
	if x.ndim > 1: return(False)
	n = inverse_choose(len(x), 2)
	return(x.ndim == 1 and n == int(n))

def is_point_cloud(x: ArrayLike) -> bool: 
	''' Checks whether 'x' is a 2-d array of points '''
	return(isinstance(x, np.ndarray) and x.ndim == 2)

def as_dist_matrix(x: ArrayLike, metric="euclidean") -> ArrayLike:
	if is_pairwise_distances(x):
		n = inverse_choose(len(x), 2)
		assert n == int(n)
		D = np.zeros(shape=(n, n))
		D[np.triu_indices(n, k=1)] = x
		D = D.T + D
		return(D)
	elif is_distance_matrix(x):
		return(x)
	else:
		return(dist(x, as_matrix=True, metric=metric))

def is_dist_like(x: ArrayLike):
	return(is_distance_matrix(x) or is_pairwise_distances(x))

def is_index_like(x: ArrayLike):
	return(np.all([isinstance(el, numbers.Integral) for el in x]))

def subset_dist(x: ArrayLike, I: Union[ArrayLike, Tuple]):
	''' 
	Used for subsetting dist-like objects
	
	Parameters: 
		x := pairwise distances or distance matrix 
		I := index vector of subset to obtain or tuple of length 2 where each element gives the subsets to choose from.

	Examples:
		d = dist(np.random.uniform(size=(100,2)))
		D = subset_dist(d, [0,1,2,5,10]) 			## vector of length 10 containing pairwise distances for subset [0,1,2,5,10]
		D = subset_dist(d, ([0,1], [2,5,10])) ## matrix of size (2, 3) containing distances from 0 and 1 to 2, 5, and 10
		
		d = dist(np.random.uniform(size=(100,2)), as_matrix = True)
		D = subset_dist(d, [0,1,2,5,10]) 			## matrix of size (5, 5) containing distances for indices [0,1,2,5,10]
		D = subset_dist(d, ([0,1], [2,5,10])) ## same as the vector example above

	'''
	from itertools import combinations
	# assert is unique
	if is_distance_matrix(x):
		if isinstance(I, Tuple) and len(I) == 2:
			subset = x[np.ix_(I[0], I[1])]
		else: 
			subset = x[np.ix_(I, I)]
	elif is_pairwise_distances(x):
		n = inverse_choose(len(x), 2)
		if isinstance(I, Tuple) and len(I) == 2:
			subset = np.zeros((len(I[0]), len(I[1])))
			for i, ind0 in enumerate(I[0]):
				for j, ind1 in enumerate(I[1]):
					if ind0 == ind1:
						subset[i,j] = 0.0
					else: 
						subset[i,j] = x[rank_comb2(ind0,ind1,n)]
		else: 
			subset = np.array([x[rank_comb2(i,j,n)] for i,j in combinations(I, 2)])
	else: 
		raise ValueError("'x' must be a distance matrix or a set of pairwise distances")
	return(subset)
			
# @numba.jit(cache=True, nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
# def numba_dist(a, b):
# 	dist = np.zeros(a.shape[0])
# 	for r in range(a.shape[0]):
# 		for c in range(128):
# 			dist[r] += (b[c] - a[r, c])**2

def dist(x: npt.ArrayLike, y: Optional[npt.ArrayLike] = None, pairwise = False, as_matrix = False, metric : Union[str, Callable] = 'euclidean', **kwargs):
	''' 
	Thin wrapper around the 'cdist' and 'pdist' functions from the scipy.spatial.distance module. 
	
	If x and y are  (n x d) and (m x d) numpy arrays, respectively, then:
	Usage:  
		(1) dist(x)                           => (n choose 2) pairwise distances in x
		(2) dist(x, as_matrix = True)         => (n x n) distance matrix
		(3) dist(x, y)                        => (n x m) distances between x and y. If x == y, then equivalent to (2).
		(4) dist(x, y, pairwise = True)       => (n x 1) individual pairwise distances between x and y (requires n == m).
		(5) dist(..., metric = "correlation") => any of (1-4) specialized with a given metric
	The supplied 'metric' can be either a string or a real-valued binary distance function. Both the 
	metric and the additional keyword arguments are passed to 'pdist' or 'cdist', respectively. 
	See for reference the scipy.spatial.distance documentation, https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist, for details.
	
	TODO: remove as_matrix, use only pairwise flag, and set default pairwise=False (simplifies (3-4) if default is false)
	'''
	x = np.asanyarray(x) ## needs to be numpy array
	if (x.shape[0] == 1 or x.ndim == 1) and y is None: 
		return(np.zeros((0, np.prod(x.shape))))
	if y is None:
		#return(cdist(x, x, metric, **kwargs) if (as_matrix) else pdist(x, metric, **kwargs))
		return(squareform(pdist(x, metric, **kwargs)) if (as_matrix) else pdist(x, metric, **kwargs))
	else:
		n, y = x.shape[0], np.asanyarray(y)
		if pairwise:
			if x.shape != y.shape: raise Exception("x and y must have same shape if pairwise=True.")
			return(np.array([cdist(x[ii:(ii+1),:], y[ii:(ii+1),:], metric, **kwargs).item() for ii in range(n)]))
		else:
			if x.ndim == 1: x = np.reshape(x, (1, len(x)))
			if y.ndim == 1: y = np.reshape(y, (1, len(y)))
			return(cdist(x, y, metric, **kwargs))


# import numba 
# def cyclic_dist(lb, ub):
# 	''' Mobius band distance '''
# 	return(1.0)


# def 
