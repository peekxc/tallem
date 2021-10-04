import numpy as np 
import numpy.typing as npt
from bisect import bisect_left
from scipy.sparse import issparse


def find_points(query: npt.ArrayLike, reference: npt.ArrayLike):
	'''
	Given two point clouds 'query' and 'reference', finds the row index of every point in 
	the query set in the reference set if it exists, or -1 otherwise. Essentially performs 
	two binary searches on the first dimension to extracts upper and lower bounds on points 
	with potentially duplicate first dimensions, and then resorts to a linear search on such 
	duplicates
	'''
	Q, R = np.array(query), np.array(reference)
	m = R.shape[0]
	lex_indices = np.lexsort(R.T[tuple(reversed(range(R.shape[1]))),:])
	lb_idx = np.searchsorted(a = R[lex_indices,0], v = Q[:,0])
	ub_idx = np.searchsorted(a = R[lex_indices,0], v = Q[:,0], side="right")
	indices = np.empty(Q.shape[0], dtype = np.int32)
	for i in range(Q.shape[0]):
		if lb_idx[i] >= m:
			indices[i] = -1
		elif lb_idx[i] == ub_idx[i]:
			check_idx = lex_indices[lb_idx[i]]
			indices[i] = check_idx if np.all(Q[i,:] == R[check_idx,:]) else -1
		elif lb_idx[i] == (ub_idx[i] - 1):
			indices[i] = lex_indices[lb_idx[i]]
		else: 
			found = np.where((R[lex_indices[lb_idx[i]:ub_idx[i]],:] == Q[i,:]).all(axis=1))
			indices[i] = -1 if len(found) == 0 else found[0][0]
	return(indices)

def find_where(a: npt.ArrayLike, b: npt.ArrayLike, validate: bool = False):
	''' Finds where each element in 'a' is positioned in array 'b', or None otherwise. '''
	a, b = np.ravel(np.array(a, copy=False)), np.ravel(np.array(b, copy=False))
	def index(a, x):
		i = bisect_left(a, x)
		return(i if i != len(a) and a[i] == x else None)
	ind = np.array([index(b,v) for v in a], dtype=object)
	if (validate and np.any(ind == None)): raise ValueError("Unable to match valid positions for input arrays!")
	if not(np.any(ind == None)):
		ind = np.array(ind, dtype=np.int32)
	return(ind)

def inverse_permutation(a):
	b = np.arange(a.shape[0])
	b[a] = b.copy()
	return b

def as_np_array(a: npt.ArrayLike) -> npt.ArrayLike:
	''' Converts sparse objects or numpy-compatible containers to dense numpy arrays '''
	if issparse(a): 
		a = a.todense()
		return(a + a.T - np.diag(a.diagonal()))
	return(np.array(a, copy=False))

# Augmented from: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(arrays):
	arrays = np.asanyarray(arrays)
	la = len(arrays)
	dtype = np.find_common_type([a.dtype for a in arrays], [])
	arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
	for i, a in enumerate(np.ix_(*arrays)):
			arr[..., i] = a
	return arr.reshape(-1, la)

import math
def inverse_choose(x: int, k: int):
	assert k >= 1, "k must be >= 1" 
	if k == 1: return(x)
	if k == 2:
		rng = np.array(list(range(int(np.floor(np.sqrt(2*x))), int(np.ceil(np.sqrt(2*x)+2) + 1))))
		final_n = rng[np.nonzero(np.array([math.comb(n, 2) for n in rng]) == x)[0].item()]
	else:
		# From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
		if x < 10**7:
			lb = (math.factorial(k)*x)**(1/k)
			potential_n = np.array(list(range(int(np.floor(lb)), int(np.ceil(lb+k)+1))))
			idx = np.nonzero(np.array([math.comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[idx]
		else:
			lb = np.floor((4**k)/(2*k + 1))
			C, n = math.factorial(k)*x, 1
			while n**k < C: n = n*2
			m = (np.nonzero( np.array(list(range(1, n+1)))**k >= C )[0])[0].item()
			potential_n = np.array(list(range(int(np.max([m, 2*k])), int(m+k+1))))
			ind = np.nonzero(np.array([math.comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[ind]
	return(final_n)
