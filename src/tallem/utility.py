import numpy as np 
import numpy.typing as npt
from bisect import bisect_left

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

	# p = np.argsort(a)
	# ind = p[np.searchsorted(a[p], b)]
	# return(np.where(a[ind] == b, ind, None))

def inverse_permutation(a):
	b = np.arange(a.shape[0])
	b[a] = b.copy()
	return b