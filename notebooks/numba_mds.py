import os
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import *
from .distance import *
from .utility import *
from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eigh, eig as dense_eig
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

import numpy as np
import numba as nb
from numba import njit, types, float32, float64, int32, int64, prange
from numba.extending import overload


@njit('float64[:,:](float64[:,:])', fastmath=True,parallel=False)
def average_rows(x):
	assert x.ndim == 2
	res = np.zeros((1, x.shape[0]),dtype=np.float64)
	for i in prange(x.shape[0]):
		res += x[i,:]
	return res / x.shape[0]

@njit('float64[:,:](float64[:,:])', fastmath=True,parallel=False)
def average_cols(x):
	assert x.ndim == 2
	res = np.zeros((1, x.shape[1]),dtype=np.float64)
	for i in prange(x.shape[1]):
		res += x[:,i]
	return res / x.shape[1]

#test.parallel_diagnostics(level=4)


@njit('float64[:,:](float64[:,:], int32)', fastmath=False)
def cmds_numba_naive(D, d):
	n = D.shape[0]
	H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
	B = -0.5 * H @ D @ H
	evals, evecs = np.linalg.eigh(B)
	evals, evecs = np.flip(evals)[np.arange(d)], np.fliplr(evecs)[:,np.arange(d)]        
	w = np.flatnonzero(evals > 0)
	Y = np.zeros(shape=(n, d))
	Y[:,w] = evecs[:,w] @ np.diag(np.sqrt(evals[w]))
	return(Y)

## Classical MDS with Numba
@njit(nb.types.Tuple((float64[:], float64[:,:]))(float64[:,:], int32), fastmath=False)
def cmds_numba_E(D, d):
	''' Given distance matrix 'D' and dimension 'd', computes the classical MDS '''
	D = -0.5*(D - average_rows(D) - average_cols(D).T + np.mean(D))
	evals, evecs = np.linalg.eigh(D)
	evals, evecs = np.flip(evals)[:d] , np.fliplr(evecs)[:,:d] 
	return((evals, evecs))

@njit('float64[:,:](float64[:,:], int32)', fastmath=False)
def cmds_numba(D, d):
	n = D.shape[0]
	evals, evecs = cmds_numba_E(D, d)
	w = np.flatnonzero(evals > 0)
	Y = np.zeros(shape=(n, d))
	Y[:,w] = np.dot(evecs[:,w], np.diag(np.sqrt(evals[w])))
	return(Y)

from tallem.syevr import numba_dsyevr

@njit('float64[:,:](float64[:,:], int32)', fastmath=False)
def cmds_numba_fortran(D, d):
	n = D.shape[0]
	D = -0.5*(D - average_rows(D) - average_cols(D).T + np.mean(D))
	evals, evecs, i, e = numba_dsyevr(D, n-d+1, n, 1e-8)
	w = np.flatnonzero(evals > 0)
	Y = np.zeros(shape=(n, d))
	Y[:,w] = np.dot(evecs[:,w], np.diag(np.sqrt(evals[w])))
	return(Y)

@njit('float64[:,:](float64[:,:], float64[:,:], int32)', fastmath=False)
def landmark_cmds_numba(LD, S, d):
	''' 
	Barbones landmark MDS with Numba 
	
	LD := (k x k) landmark distance matrix 
	S := (k x n) matrix of distances from the n points to the k landmark points, where n > k
	d := dimension of output coordinitization
	'''
	n = S.shape[1]
	evals, evecs = cmds_numba_E(LD, d)
	mean_landmark = average_cols(LD).T
	w = np.flatnonzero(evals > 0)
	L_pseudo = evecs/np.sqrt(evals[w])
	Y = np.zeros(shape=(n, d))
	Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 
	return(Y)

# lapack.dsyevr(jobz, rng, uplo, N, D, N, vl, vu, il, iu, tol, m, w, Z, ldz, isuppz, work, lwork, iwork, liwork, info)
@njit('float64[:,:](float64[:,:])', parallel=True)
def dist_matrix(X):
	n = X.shape[0]
	D = np.zeros((n,n))
	for i in np.arange(n):
		for j in np.arange(n):
			D[i,j] = np.sum((X[i,:]-X[j,:])**2)
	return(D)

# @njit('float64[:,:](float64[:,:], int32[:], int32[:], int32)', parallel=True)
# def bench_parallel(X, subsets_vec, subsets_len, d):
# 	results = []
# 	for i in prange(len(subsets_vec)-1):
# 		ind = np.arange(np.subsets_vec[i], subsets_vec[i+1])
# 		D = dist_matrix(X[ind,:])
# 		results.append(cmds_numba(D, d))
# 	return(results)

#from numba import njit, prange
# @njit(parallel=True)
# def fit_local_models(f, X, cover):
# 	index_set = list(cover.keys())
# 	subsets = list(cover.values())
# 	result = {}
# 	for j in prange(len(cover)):
# 		index, subset = index_set[j], subsets[j]
# 		result[index] = f(X[np.array(subset),:])
# 	return(result)
