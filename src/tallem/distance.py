# %% Distance imports
import numpy as np
import numpy.typing as npt 
from typing import Callable, Optional, Union
from scipy.spatial.distance import pdist, cdist
from scipy.sparse import issparse

# %% Distance definitions
def is_distance_matrix(x: npt.ArrayLike) -> bool:
	x = np.array(x, copy=False)
	is_square = x.ndim == 2	and (x.shape[0] == x.shape[1])
	return(False if not(is_square) else np.all(np.diag(x) == 0))

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
	'''
	x = np.array(x, copy=False) ## needs to be numpy array
	if x.shape[0] == 1 or x.ndim == 1 and y is None: 
		return(np.zeros((0, np.prod(x.shape))))
	if y is None:
		return(cdist(x, x, metric, **kwargs) if (as_matrix) else pdist(x, metric, **kwargs))
	else:
		n, y = x.shape[0], np.array(y, copy=False)
		if pairwise:
			if x.shape != y.shape: raise Exception("x and y must have same shape if pairwise=True.")
			return(np.array([cdist(x[ii:(ii+1),:], y[ii:(ii+1),:], metric, **kwargs).item() for ii in range(n)]))
		else:
			if x.ndim == 1: x = np.reshape(x, (len(x), 1))
			if y.ndim == 1: y = np.reshape(y, (len(y), 1))
			return(cdist(x, y, metric, **kwargs))
