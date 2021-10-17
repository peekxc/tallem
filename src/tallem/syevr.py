import ctypes
import numpy as np
import numba as nb
from numba import vectorize, njit, jit
from numba.extending import get_cython_function_address

addr = get_cython_function_address('scipy.linalg.cython_lapack','dsyevr')
_PTR, _dble, _char, _int  = ctypes.POINTER, ctypes.c_double, ctypes.c_char, ctypes.c_int
_ptr_dble = _PTR(_dble)
_ptr_char = _PTR(_char)
_ptr_int  = _PTR(_int)

## Retrieve function address
functype = ctypes.CFUNCTYPE(None,
	_ptr_int, # JOBVS
	_ptr_int, # RANGE
	_ptr_int, # UPLO
	_ptr_int, # N
	_ptr_dble, # A
	_ptr_int, # LDA
	_ptr_dble, # VL
	_ptr_dble, # VU
	_ptr_int, # IL
	_ptr_int, # IU
	_ptr_dble, # ABSTOL
	_ptr_int, # M
	_ptr_dble, # W
	_ptr_dble, # Z
	_ptr_int, # LDZ
	_ptr_int, # ISUPPZ 
	_ptr_dble, # WORK
	_ptr_int, # LWORK
	_ptr_int, # IWORK 
	_ptr_int, # LIWORK 
	_ptr_int, # INFO
)
dsyevr_fn = functype(addr)

_ORD_JOBVS = ord('V')
_ORD_RNG = ord('I')
_ORD_UPLO = ord('U')

@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:,:], nb.int32, nb.int32))(nb.float64[:,:], nb.int32, nb.int32, nb.float32), nopython=True, parallel=True, fastmath=True)
def numba_dsyevr(x, I1, I2, tolerance):
	''' Computes all eigenvalues in range 1 <= I1 <= I2 <= N'''
	assert (x.shape[0] == x.shape[1])
	assert I1 <= I2 and I1 >= 1 and I2 <= x.shape[0]
	
	## Setup arguments
	i1, i2 = I1, I2
	n, m = x.shape[0], abs(i2-i1)+1
	JOBVS = np.array([_ORD_JOBVS], np.int32)
	RNG = np.array([_ORD_RNG], np.int32)
	UPLO = np.array([_ORD_UPLO], np.int32)
	N = np.array([n], np.int32)
	A = x.copy()     # in & out
	LDA = np.array([n], np.int32)
	VL, VU =  np.array(0, np.float64), np.array(0, np.float64), 
	IL, IU = np.array(i1, np.int32), np.array(i2, np.int32)
	ABS_TOL = np.array([tolerance], np.float64)
	N_EVALS_FOUND = np.empty(1, np.int32)
	W = np.zeros(n, np.float64)
	LDZ, ISUPPZ = np.array([n], np.int32), np.zeros(2*m, np.int32) 
	INFO = np.array([0.0], np.int32)

	## Output
	Z = np.zeros((m, n), np.float64).T
	
	# preallocate
	LWORK, LIWORK = np.array([26*n], np.int32), np.array([10*n], np.int32)
	WORK, IWORK = np.zeros(26*n, np.float64), np.zeros(10*n, np.int32)

	## Dry-run to get workspace allocation amount 
	# LWORK, LIWORK = np.array([-1.0], np.int32), np.array([-1.0], np.int32)
	# WORK, IWORK = np.zeros(1, np.float64), np.zeros(1, np.int32)
	# dsyevr_fn(
	# 	JOBVS.ctypes,RNG.ctypes,UPLO.ctypes,N.ctypes,A.view(np.float64).ctypes,
	# 	LDA.ctypes,VL.ctypes,VU.ctypes,IL.ctypes,IU.ctypes,ABS_TOL.ctypes,N_EVALS_FOUND.ctypes,
	# 	W.view(np.float64).ctypes, Z.view(np.float64).ctypes,
	# 	LDZ.ctypes,ISUPPZ.view(np.int32).ctypes,WORK.view(np.float64).ctypes,
	# 	LWORK.ctypes,IWORK.view(np.int32).ctypes,LIWORK.ctypes,INFO.ctypes
	# )
	# LWORK, LIWORK = np.array([WORK[0]], np.int32), np.array([IWORK[0]], np.int32)
	# WORK, IWORK = np.array([LWORK[0]], np.float64), np.array([LIWORK[0]], np.int32)

	## Call the function
	dsyevr_fn(
		JOBVS.ctypes,
		RNG.ctypes,
		UPLO.ctypes,
		N.ctypes,
		A.view(np.float64).ctypes,
		LDA.ctypes,
		VL.ctypes,
		VU.ctypes,
		IL.ctypes,
		IU.ctypes,
		ABS_TOL.ctypes,
		N_EVALS_FOUND.ctypes,
		W.view(np.float64).ctypes,
		Z.view(np.float64).ctypes,
		LDZ.ctypes,
		ISUPPZ.view(np.int32).ctypes,
		WORK.view(np.float64).ctypes,
		LWORK.ctypes,
		IWORK.view(np.int32).ctypes,
		LIWORK.ctypes,
		INFO.ctypes
	)
	return((W[:m], Z, INFO[0], N_EVALS_FOUND[0]))