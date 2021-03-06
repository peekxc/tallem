# cython: profile=True, linetrace=True, binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1, language = c++

import cython
import numpy as np
from typing import * 
from numpy.typing import ArrayLike
from cython cimport view
from cpython.array cimport array, clone
from cython.parallel import prange
from scipy.linalg.cython_lapack cimport dsyevr
from libc.math cimport sqrt

# from scipy.spatial.distance import squareform, pdist # for validation

from libcpp.vector cimport vector

## TODO: try this in C++ from pybind! 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef public void cython_dsyevr_cpp(double* A, double* W, double* Z, int n, int IL, int IU, double tolerance):
	''' 
	Computes all eigenvalues/vectors in range 1 <= IL <= IU <= N of an (N x N) real symmetric matrix 'x' 

	Calls underlying LAPACK procedure 'dyevr'

	Returns a tuple (evals, evecs) where: 
		evals := a (d,)-shaped array of requested eigenvalues, in ascending order, where d = IU-IL+1
		evecs := a (n, d)-shaped array of the requested eigenvectors corresponding in order to 'evals'
	'''
	cdef int N = n
	cdef int M = IU-IL+1
	cdef char* JOBVS = 'V'
	cdef char* RNG = 'I'
	cdef char* UPLO = 'U'
	cdef int LDA = N
	cdef int LDZ = N
	cdef double VL = 0.0 
	cdef double VU = 0.0
	cdef int INFO = 0
	cdef int N_EVALS_FOUND = 0
	cdef int LIWORK = 10*n
	cdef int LWORK = 26*n
	cdef vector[int] IWORK
	cdef vector[double] WORK
	cdef vector[int] ISUPPZ
	IWORK.resize(LIWORK)
	WORK.resize(LWORK)
	ISUPPZ.resize(2*M)
	dsyevr(
		JOBVS,RNG,UPLO,&N,A,
		&LDA,&VL,&VU,&IL,&IU,&tolerance,
		&N_EVALS_FOUND,W,Z,
		&LDZ, ISUPPZ.data(),WORK.data(),&LWORK,IWORK.data(),&LIWORK,&INFO
	)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef public void _dyevr(double[::1, :] A, int IL, int IU, double ABS_TOL, double[:] W, double[:] WORK, int[:] IWORK, int[:] ISUPPZ, double[::1, :] Z):
	cdef int N = A.shape[0]
	cdef int M = IU-IL+1
	cdef char* JOBVS = 'V'
	cdef char* RNG = 'I'
	cdef char* UPLO = 'U'
	cdef int LDA = N
	cdef int LDZ = N
	cdef double VL = 0.0 
	cdef double VU = 0.0
	cdef int INFO = 0
	cdef int N_EVALS_FOUND = 0
	cdef int LIWORK = IWORK.size
	cdef int LWORK = WORK.size
	dsyevr(
		JOBVS,RNG,UPLO,&N,&A[0,0],
		&LDA,&VL,&VU,&IL,&IU,&ABS_TOL,
		&N_EVALS_FOUND,&W[0],&Z[0,0],
		&LDZ, &ISUPPZ[0],&WORK[0],&LWORK,&IWORK[0],&LIWORK,&INFO
	)


cpdef cython_dsyevr_inplace(double[::1,:] D, IL, IU, tolerance, int n, double[:] W, double[::1,:] Z):
	''' 
	Computes all eigenvalues/vectors in range 1 <= IL <= IU <= N of an (N x N) real symmetric matrix 'D' 

	Calls underlying LAPACK procedure 'dyevr'

	Returns a tuple (evals, evecs) where: 
		evals := a (d,)-shaped array of requested eigenvalues, in ascending order, where d = IU-IL+1
		evecs := a (n, d)-shaped array of the requested eigenvectors corresponding in order to 'evals'

	Notes:
		- D will be overriden here!
	'''
	assert (D.shape[0] == D.shape[1])
	assert IL <= IU and IL >= 1 and IU <= D.shape[0]
	cdef int m = abs(IU-IL)+1
	# cdef double[:] W = np.empty((n,), np.float64)
	cdef double[:] WORK = np.empty((26*n,), np.float64)
	cdef int[:] ISUPPZ = np.empty((2*m,), np.int32) 
	cdef int[:] IWORK = np.empty((10*n,), np.int32)
	_dyevr(D, IL, IU, tolerance, W, WORK, IWORK, ISUPPZ, Z)

cpdef public cython_dsyevr(x: ArrayLike, IL: int, IU: int, tolerance: float, copy: bool):
	''' 
	Computes all eigenvalues/vectors in range 1 <= IL <= IU <= N of an (N x N) real symmetric matrix 'x' 

	Calls underlying LAPACK procedure 'dyevr'

	Returns a tuple (evals, evecs) where: 
		evals := a (d,)-shaped array of requested eigenvalues, in ascending order, where d = IU-IL+1
		evecs := a (n, d)-shaped array of the requested eigenvectors corresponding in order to 'evals'
	'''
	assert (x.shape[0] == x.shape[1])
	assert IL <= IU and IL >= 1 and IU <= x.shape[0]
	if not((x.flags["F_CONTIGUOUS"]) and (x.dtype == np.float64)):
		x = x.astype(dtype=np.float64, order='F', copy=True)
		copy = False
	if copy: 
		x = x.copy()
	n, m = x.shape[0], abs(IU-IL)+1
	W = np.empty((n,), np.float64)
	ISUPPZ = np.empty((2*m,), np.int32) 
	Z = np.zeros((n, m), dtype=np.float64, order='F')
	WORK, IWORK = np.empty((26*n,), np.float64), np.empty((10*n,), np.int32)
	_dyevr(x, IL, IU, tolerance, W, WORK, IWORK, ISUPPZ, Z)
	return((W[:m], Z))


## Applies double-centering to a square matrix 'D'
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double_center(double[::1, :] D, int n):
	cdef int i, j 
	cdef double total = 0.0
	cdef double[:] row_sum_view = np.zeros((n,), dtype=np.float64)
	cdef double[:] col_sum_view = np.zeros((n,), dtype=np.float64)
	# cdef double[::1] row_sum_view = cnp.empty(n, dtype=np.float64)
	# cdef double[::1] col_sum_view = cnp.empty(n, dtype=np.float64)
	# for i in range(n):
	# 	row_sum_view[i] = 0.0
	# 	col_sum_view[i] = 0.0
	for i in range(n):
		for j in range(n):
			row_sum_view[i] += D[i,j]
			col_sum_view[j] += D[i,j]
			total += D[i,j]
	for i in range(n):
		row_sum_view[i] /= n
		col_sum_view[i] /= n
	total /= (n*n)
	for i in range(n):
		for j in range(n):
			D[i,j] = -0.5*(D[i,j] - row_sum_view[i] - col_sum_view[j] + total)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cython_cmds_fortran_inplace(double[::1, :] D, int d, int n, double[::1, :] Y, int offset):
	double_center(D, n) # double-center first (n x n submatrix of D)
	cdef double[:] W = np.empty(n, dtype=np.float64)
	cdef double[::1,:] Z = np.zeros(shape=(n, d), dtype=np.float64, order='F')
	cdef double c_eval = 0.0
	cdef int i, di
	cython_dsyevr_inplace(D, n-d+1, n, 1e-8, n, W, Z)
	for di in range(d):
		if W[di] > 0:
			c_eval = sqrt(W[di])
			for i in range(n):
				Y[di, offset+i] = c_eval*Z[i,di]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dist_matrix_subset(const double[::1, :] X, const int[:] ind, double[::1, :] D):
	''' 
	Computes squared euclidean distance matrix of the points given by X[ind,:], 
	storing the results in the first (n x n) submatrix of D 
	'''
	cdef Py_ssize_t n = ind.size
	cdef Py_ssize_t d = X.shape[1]
	cdef Py_ssize_t i, j, k
	cdef double tmp, diff
	for i in prange(n, nogil=True, schedule='guided'):
		for j in range(i+1,n):
			tmp = 0.0
			for k in range(d):
				diff = X[ind[i], k] - X[ind[j], k]
				tmp = tmp + (diff * diff)
			D[i,j] = tmp
			D[j,i] = tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_cmds_parallel(const double[::1, :] X, const int d, const int[:] ind_vec, const int[:] ind_len, const int max_n, double[::1, :] output):
	''' 
		X := (d,n) matrix [columns-oriented (Fortran-style)] of points 
		ind_vec := (m,) contiguous vector of indices for each subset 
		ind_len := (j+1,) contiguous vector such that ind_vec[ind_len[i]:ind_len[i+1]] represents the i'th subset
	'''
	cdef int N = len(ind_vec)
	assert output.shape[0] == d and output.shape[1] == N
	cdef double[::1, :] D_reuse = np.zeros((max_n,max_n), dtype='double', order='F')
	cdef int i, ni, nj, local_n
	cdef int M = len(ind_len)-1
	for i in range(M):
		ni = ind_len[i]
		nj = ind_len[i+1]
		local_n = nj - ni
		dist_matrix_subset(X, ind_vec[ni:nj], D_reuse)
		cython_cmds_fortran_inplace(D_reuse, d, local_n, output, ni)

def flatten_list_of_lists(lst_of_lsts):
	N = sum(map(len, lst_of_lsts))  # number of elements in the flattened array   
	starts = np.empty(len(lst_of_lsts)+1, dtype=np.int32)  # needs place for one sentinel
	values = np.empty(N, dtype=np.int32)
	starts[0], cnt = 0, 0
	for i,lst in enumerate(lst_of_lsts):
		for el in lst:
			values[cnt] = el
			cnt += 1       # update index in the flattened array for the next element
		starts[i+1] = cnt  # remember the start of the next list
	return values, starts

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pdist_matrix_subset(const double[::1, :] X, const int[:] s, const int[:] t, double[::1, :] D):
	'''
	Computes (l x n) distances, stored in D
	Parameters: 
		X := (d x n)
		s := (k,)-sized vector of indices 
		t := (m,)-sized vector of indices
		D := (k*,m*)-sized matrix to store the output 
	'''
	cdef int n = X.shape[1]
	cdef int d = X.shape[0]
	cdef int k = s.size
	cdef int m = t.size
	for i in range(k):
		for j in range(m):
			tmp = 0.0
			for di in range(d):
				diff = X[s[i], di] - X[t[j], di]
				tmp = tmp + (diff * diff)
			D[i,j] = tmp
		

# from tallem.extensions import landmark

# cpdef cython_landmark_mds(const double[::1, :] X, const int d, const int[:] ind_vec, const int[:] ind_len, const int max_n, double[::1, :] output):
# 	''' 
# 	Barbones landmark MDS with Cython 

# 	Parameters: 
# 		X := (D,n) matrix [column-major] of points 
# 		d := target dimension for mds 
# 		ind_vec := flat (m,)-sized contiguous vector of indices for all subsets
# 		ind_len := flat (j+1,)-sized contiguous vector satisfying ind_vec[ind_len[i]:ind_len[i+1]] represents the i'th subset
# 		max_n := maximum size of any given cover subset
# 		output := (d, m) matrix [column-major] to store the output embeddings
# 	'''
# 	cdef int N = len(ind_vec)
# 	assert output.shape[0] == d and output.shape[1] == N
# 	cdef int k = d*3
# 	cdef double[::1, :] LD_reuse = np.zeros((k,k), dtype=np.float64, order='F') # landmark distance matrix 
# 	cdef double[::1, :] S_reuse = np.zeros((k, max_n), dtype=np.float64, order='F')
# 	cdef int i, j, ni, nj, local_n
# 	cdef int J = len(ind_len)-1 # number of subsets
# 	for j in range(J):
# 		pdist_matrix_subset( S_reuse)
# 		a = X[]
		# indices, radii = landmark.maxmin(a.T, eps, k, False, seed)

	# n = S.shape[1]
	# evals, evecs = cmds_numba_E(LD, d)
	# mean_landmark = average_cols(LD).T
	# w = np.flatnonzero(evals > 0)
	# L_pseudo = evecs/np.sqrt(evals[w])
	# Y = np.zeros(shape=(n, d))
	# Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 
