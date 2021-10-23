import cython
import numpy as np

from cython cimport view
from cpython.array cimport array, clone
from cython.parallel import prange
from scipy.linalg.cython_lapack cimport dsyevr

cpdef void _dyevr(double[::1, :] A, int IL, int IU, double ABS_TOL, double[:] W, double[:] WORK, int[:] IWORK, int[:] ISUPPZ, double[::1, :] Z):
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

def cython_dsyevr(x, IL, IU, tolerance):
	''' 
	Computes all eigenvalues/vectors in range 1 <= IL <= IU <= N of an (N x N) real symmetric matrix 'x' 

	Calls underlying LAPACK procedure 'dyevr'
	'''
	assert (x.shape[0] == x.shape[1])
	assert IL <= IU and IL >= 1 and IU <= x.shape[0]
	x = np.asfortranarray(x.copy())
	n, m = x.shape[0], abs(IU-IL)+1
	W = np.empty((n,), np.float64)
	ISUPPZ = np.empty((2*m,), np.int32) 
	Z = np.zeros((n, m), dtype=np.float64, order='F')
	WORK, IWORK = np.empty((26*n,), np.float64), np.empty((10*n,), np.int32)
	_dyevr(x, IL, IU, tolerance, W, WORK, IWORK, ISUPPZ, Z)
	return((W[:m], Z))

from cython cimport view
cimport numpy as cnp

# my_array = view.array(shape=(10, 2), itemsize=sizeof(int), format="i")
# cdef int[:, :] my_slice = my_array
# row_sum = view.array(shape=(n,), itemsize=sizeof(double), format="d")
# col_sum = view.array(shape=(n,), itemsize=sizeof(double), format="d")
# cdef double[:] row_sum_view = row_sum
# cdef double[:] col_sum_view = col_sum
# cdef double[:] col_sum = cnp.zeros(n, dtype=numpy.int32)

## Applies double-centering to a square matrix 'D'
cdef center(double[::1, :] D):
	cdef int i, j 
	cdef int n = D.shape[0]
	cdef double total = 0.0
	cdef double[::1] row_sum_view = np.zeros((n,), dtype='double')
	cdef double[::1] col_sum_view = np.zeros((n,), dtype='double')
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

cpdef cython_cmds_fortran(double[::1, :] D, int d, double[::1, :] Y):
	''' Puts the results into Y '''
	cdef int n = D.shape[0]
	center(D)
	evals, evecs = cython_dsyevr(D, n-d+1, n, 1e-8)
	w = np.flatnonzero(evals > 0)
	Y = np.dot(evecs[:,w], np.diag(np.sqrt(evals[w])))
	# Y = np.zeros(shape=(n, d))
	# Y[:,w] = np.dot(evecs[:,w], np.diag(np.sqrt(evals[w])))
	# return(Y)



# cpdef cython_cmds_fortran(double[::1, :] D, double[::1, :] LD, int d):
# 	''' 
# 	Barbones landmark MDS with Numba 
	
# 	LD := (k x k) landmark distance matrix 
# 	S := (k x n) matrix of distances from the n points to the k landmark points, where n > k
# 	d := dimension of output coordinitization
# 	'''
# 	cdef int k = S.shape[0]
# 	cdef int n = S.shape[1]
# 	evals, evecs = cython_dsyevr(D, n-d+1, n, 1e-8)

# 	mean_landmark = average_cols(LD).T

# 	w = np.flatnonzero(evals > 0)
# 	L_pseudo = evecs/np.sqrt(evals[w])
	
# 	Y = np.zeros(shape=(n, d))
# 	Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 

# 	cdef int n = D.shape[0]
# 	center(D)
# 	evals, evecs = cython_dsyevr(D, n-d+1, n, 1e-8)
# 	w = np.flatnonzero(evals > 0)
# 	Y = np.zeros(shape=(n, d))
# 	Y[:,w] = np.dot(evecs[:,w], np.diag(np.sqrt(evals[w])))
# 	return(Y)

	# def landmark_cmds_numba(LD, S, d):
	
	# 	n = S.shape[1]
	# 	evals, evecs = cmds_numba_E(LD, d)
	# 	mean_landmark = average_cols(LD).T
	# 	w = np.flatnonzero(evals > 0)
	# 	L_pseudo = evecs/np.sqrt(evals[w])
	# 	Y = np.zeros(shape=(n, d))
	# 	Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 
	# 	return(Y)

from scipy.spatial.distance import squareform, pdist

def cython_cmds_parallel(const double[::1, :] X, int d, int[:] ind_vec, int[:] ind_len):
	''' 
		X := (d,n) matrix [columns-oriented (Fortran-style)] of points 
		ind_vec := (m,) contiguous vector of indices for each subset 
		ind_len := (j+1,) contiguous vector such that ind_vec[ind_len[i]:ind_len[i+1]] represents the i'th subset
	'''
	cdef int N = len(ind_vec)
	cdef double[::1, :] D
	cdef double[::1, :] results = np.zeros((N,d), dtype='double')
	cdef double[::1, :] output 
	for i in range(len(ind_len)-1):
		ind = ind_vec[ind_len[i]:ind_len[i+1]]
		D = squareform(pdist(X[:,ind]))
		output = results[ind_len[i]:ind_len[i+1],:]
		cython_cmds_fortran(D, d, output)
	return(results)	


import numpy as np
def flatten_list_of_lists(lst_of_lsts):
	N = sum(map(len, lst_of_lsts))  # number of elements in the flattened array   
	starts = np.empty(len(lst_of_lsts)+1, dtype=np.uint64)  # needs place for one sentinel
	values = np.empty(N, dtype=np.int64)

	starts[0], cnt = 0, 0
	for i,lst in enumerate(lst_of_lsts):
		for el in lst:
			values[cnt] = el
			cnt += 1       # update index in the flattened array for the next element
		starts[i+1] = cnt  # remember the start of the next list

	return starts, values





# DTYPE = np.int
# ctypedef np.int_t DTYPE_t
# np.ndarray[np.float64_t, ndim=2]
# cdef const double[:] myslice   # const item type => read-only view

# def fast_cmds(x, i1, i2, tol=1e-7):
# 	cdef double[:,:] x_buffer = x
# 	cdef int I1 = i1
# 	cdef int I2 = i2
# 	cdef double eps = tol
# 			# LWORK, LIWORK = np.array([26*n], np.int32), np.array([10*n], np.int32)
# 	# WORK, IWORK = np.zeros(26*n, np.float64), np.zeros(10*n, np.int32)
# 	_fast_cmds(x_buffer, I1, I2, eps)

# def _fast_cmds(double[:,:] x, int I1, int I2, double tolerance=1e-7):
# 	return(0.0)
	# LDZ, ISUPPZ = np.array([n], np.int32), np.zeros(2*m, np.int32) 
	# INFO = np.array([0.0], np.int32)

	# ## Output
	# Z = np.zeros((m, n), np.float64).T
	
	# # preallocate
	# LWORK, LIWORK = np.array([26*n], np.int32), np.array([10*n], np.int32)
	# WORK, IWORK = np.zeros(26*n, np.float64), np.zeros(10*n, np.int32)
	# cdef double[::1] W = clone(array('d'), n, False)
	# cdef W = np.empty((n,), dtype='double')
	# LDA = np.array([n], np.int32)
	# VL, VU =  np.array(0, np.float64), np.array(0, np.float64), 
	# IL, IU = np.array(i1, np.int32), np.array(i2, np.int32)
	# ABS_TOL = np.array([tolerance], np.float64)
	# N_EVALS_FOUND = np.empty(1, np.int32)
	# W = np.zeros(n, np.float64)
	# LDZ, ISUPPZ = np.array([n], np.int32), np.zeros(2*m, np.int32) 
	# INFO = np.array([0.0], np.int32)



# def fast_cmds(double[:,:] x, int I1, int I2, double tolerance=1e-7):
# 	cdef int n = x.shape[0]
# 	cdef int m = abs(I2-I1)+1
# 	cdef char* JOBVS = 'V'
# 	return(0)
	# cdef long N = A.shape[0]
	# cdef long LWORK = A.shape[1]
	# cdef long LIWORK = A.shape[1]
	# cdef int INFO = 0
	# cdef long IA = 0 #the row index in the global array A indicating the first row of sub( A )
	# cdef long JA = 0 #The column index in the global array A indicating the first column of sub( A ).
	# cdef double[::1] WORK = np.empty(LWORK, dtype=np.float64)
	# cdef int[::1] IWORK = np.empty(LIWORK, dtype=np.int32)
	# cdef int[::1] IPIV = np.empty(N, dtype=np.int32)
	# cdef int[::1] DESCA = np.empty(N, dtype=np.int32)
	
	# JOBVS = np.array([_ORD_JOBVS], np.int32)
	# RNG = np.array([_ORD_RNG], np.int32)
	# UPLO = np.array([_ORD_UPLO], np.int32)
	# N = np.array([n], np.int32)
	# A = x.copy()     # in & out
	# LDA = np.array([n], np.int32)
	# VL, VU =  np.array(0, np.float64), np.array(0, np.float64), 
	# IL, IU = np.array(i1, np.int32), np.array(i2, np.int32)
	# ABS_TOL = np.array([tolerance], np.float64)
	# N_EVALS_FOUND = np.empty(1, np.int32)
	# W = np.zeros(n, np.float64)
	# LDZ, ISUPPZ = np.array([n], np.int32), np.zeros(2*m, np.int32) 
	# INFO = np.array([0.0], np.int32)

	## Output
	# Z = np.zeros((m, n), np.float64).T
	
	# # preallocate
	# LWORK, LIWORK = np.array([26*n], np.int32), np.array([10*n], np.int32)
	# WORK, IWORK = np.zeros(26*n, np.float64), np.zeros(10*n, np.int32)
	# dsyevr(&JOBVS,
	# 	RNG.ctypes,
	# 	UPLO.ctypes,
	# 	N.ctypes,
	# 	A.view(np.float64).ctypes,
	# 	LDA.ctypes,
	# 	VL.ctypes,
	# 	VU.ctypes,
	# 	IL.ctypes,
	# 	IU.ctypes,
	# 	ABS_TOL.ctypes,
	# 	N_EVALS_FOUND.ctypes,
	# 	W.view(np.float64).ctypes,
	# 	Z.view(np.float64).ctypes,
	# 	LDZ.ctypes,
	# 	ISUPPZ.view(np.int32).ctypes,
	# 	WORK.view(np.float64).ctypes,
	# 	LWORK.ctypes,
	# 	IWORK.view(np.int32).ctypes,
	# 	LIWORK.ctypes,
	# 	INFO.ctypes)