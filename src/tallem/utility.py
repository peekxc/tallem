import numpy as np 
import numpy.typing as npt
from bisect import bisect_left

def find_where(a: npt.ArrayLike, b: npt.ArrayLike):
	''' Finds where each element in 'a' is positioned in array 'b', or None otherwise. '''
	a, b = np.ravel(np.array(a, copy=False)), np.ravel(np.array(b, copy=False))
	def index(a, x):
		i = bisect_left(a, x)
		return(i if i != len(a) and a[i] == x else None)
	return(np.array([index(b,v) for v in a], dtype=object))
	# p = np.argsort(a)
	# ind = p[np.searchsorted(a[p], b)]
	# return(np.where(a[ind] == b, ind, None))

def inverse_permutation(a):
	b = np.arange(a.shape[0])
	b[a] = b.copy()
	return b