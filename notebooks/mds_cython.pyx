import cython
import numpy as np
cimport numpy as np

from cython cimport view
from cpython.array cimport array, clone
from cython.view cimport array as cvarray
from cython.parallel import prange
from scipy.linalg.cython_lapack cimport dsyevr
from libc.stdlib cimport malloc, free

from scipy.linalg.cython_lapack cimport dgetri, dgetrf, dpotrf

def chol_c(double[:, ::1] A, double[:, ::1] B):
	'''cholesky factorization of real symmetric positive definite float matrix A

	Parameters
	----------
	A : memoryview (numpy array)
			n x n matrix to compute cholesky decomposition
	B : memoryview (numpy array)
			n x n matrix to use within function, will be modified
			in place to become cholesky decomposition of A. works
			similar to np.linalg.cholesky
	'''
	cdef int n = A.shape[0], info
	cdef char* uplo = 'U'
	B[...] = A
	dpotrf(uplo, &n, &B[0,0], &n, &info)
	cdef int i, j
	for i in range(n):
		for j in range(n):
			if j > i:
				B[i, j] = 0 

def _dyevr(double[::1, :] A, int IL, int IU, double ABS_TOL, double[:] W, double[:] WORK, int[:] IWORK, int[:] ISUPPZ, double[::1, :] Z):
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
		JOBVS,RNG,UPLO,
		&N,
		&A[0,0],
		&LDA,&VL,&VU,&IL,&IU,&ABS_TOL,&N_EVALS_FOUND,
		&W[0],
		&Z[0,0],&LDZ, 
		&ISUPPZ[0],
		&WORK[0],	&LWORK,
		&IWORK[0],&LIWORK,
		&INFO
	)

def cython_dsyevr(x, IL, IU, tolerance):
	''' Computes all eigenvalues in range 1 <= IL <= IU <= N'''
	assert (x.shape[0] == x.shape[1])
	assert IL <= IU and IL >= 1 and IU <= x.shape[0]
	x = np.asfortranarray(x.copy())
	n, m = x.shape[0], abs(IU-IL)+1
	W = np.empty((n,), np.float64)
	ISUPPZ = np.empty((2*m,), np.int32) 
	Z = np.zeros((n, m), dtype=np.float64, order='F')
	WORK, IWORK = np.empty((26*n,), np.float64), np.empty((10*n,), np.int32)
	_dyevr(x, IL, IU, tolerance, W, WORK, IWORK, ISUPPZ, Z)
	return((W, Z))

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
	# 	INFO.ctypes
	# return(dsyevr(a=x, n=x.shape[0], jobz='V', range='I', uplo='U', il=I1, iu=I2, abstol=tolerance))



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