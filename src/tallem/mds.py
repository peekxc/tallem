# %% MDS imports 
import numpy as np
import numpy.typing as npt
from .distance import dist
from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eig as dense_eig

# %% MDS definitions
def sammon(data, k: int = 2, max_iterations: int = 250, max_halves: int = 10):
	"""
	This implementation is adapted from a (1) GSOC 2016 project by Daniel McNeela (1) 
	which itself is adapted from the (2) Matlab implementation by Dr. Gavin C. Cawley.
	
	Sources can be found here: 
	1. https://github.com/mcneela/Retina
	2. https://people.sc.fsu.edu/~jburkardt/m_src/profile/sammon_test.m
	"""
	TolFun = 1 * 10 ** (-9)
	
	D = dist(data, as_matrix = True)
	N = data.shape[0]
	scale = np.sum(D.flatten('F'))
	D = D + np.identity(N)
	D_inv = np.linalg.inv(D)
	
	y = np.random.randn(N, k)
	one = np.ones((N, k))
	d = dist(y, as_matrix = True) + np.identity(N)
	d_inv = np.linalg.inv(d)
	delta = D - d
	E = np.sum(np.sum(np.power(delta, 2) * D_inv))

	for i in range(max_iterations):
		delta = d_inv - D_inv
		deltaone = np.dot(delta, one)
		g = np.dot(delta, y) - y * deltaone
		dinv3 = np.power(d_inv, 3)
		y2 = np.power(y, 2)
		H = np.dot(dinv3, y2) - deltaone - 2 * np.multiply(y, np.dot(dinv3, y)) + np.multiply(y2, np.dot(dinv3, one))
		s = np.divide(-np.transpose(g.flatten('F')), np.transpose(np.abs(H.flatten('F'))))
		y_old = y

	for j in range(max_halves):
		[rows, columns] = y.shape
		y = y_old.flatten('F') + s
		y = y.reshape(rows, columns)
		d = dist(y, as_matrix = True) + np.identity(N)
		d_inv = np.linalg.inv(d)
		delta = D - d
		E_new = np.sum(np.sum(np.power(delta, 2) * D_inv))
		if E_new < E:
			break
		else:
			s = 0.5 * s

	E = E_new
	E = E * scale
	return (y, E)

## Classical MDS 
def classical_MDS(a: npt.ArrayLike, k: np.int32 = 2, coords: bool = True):
	''' computes cmdscale(a) w/ a being a distance matrix '''
	a = np.array(a, copy=False)
	n, d = a.shape
	if n <= 1: return(np.repeat(0.0, d))
	if (n != d): raise ValueError("Expecting square distance matrix.")
	C = np.eye(n) - (1.0/n)*np.ones(shape=(n,n))
	B = (-1/2)*(C @ a @ C)
	if k >= (n-1):
		E = dense_eig(B)
		if k == (n - 1):
			max_idx = np.argsort(-E[0])[:(n-1)]
			E = ([E[0][max_idx], E[1][:,max_idx]]) 
	else: 
		E = truncated_eig(B, k = k)
	## Eigenvalues should be real since matrix was symmetric 
	E = (np.real(E[0]), np.real(E[1]))
	return(E[0] * E[1] if coords else E)
