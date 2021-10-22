import numpy as np
import numba as nb
from numba import njit, types, float32, float64, int32, int64, prange
from numba.extending import overload
from numba.pycc import CC

cc = CC('dr')
cc.verbose = True

@cc.export('cmds_cc', 'float64[:,:](float64[:,:], int32)')
def cmds_cc(D, d):
	n = D.shape[0]
	# D = -0.5*(D - average_rows(D) - average_cols(D).T + np.mean(D))
	evals, evecs, i, e = numba_dsyevr(D, n-d+1, n, 1e-8)
	w = np.flatnonzero(evals > 0)
	Y = np.zeros(shape=(n, d))
	Y[:,w] = np.dot(evecs[:,w], np.diag(np.sqrt(evals[w])))
	return(Y)