# %% Ensure regular MDS is working
import numpy as np
from tallem.distance import dist
from tallem.dimred import cmds
from tallem.pbm import landmark

n = 25
grid_x, grid_y = np.meshgrid(range(n), range(n))
X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)
	
Z = cmds(X, 2, coords=True)
Y = landmark.cmds(dist(X, as_matrix=True)**2, 2)

import matplotlib.pyplot as plt
plt.scatter(*Z.T)
plt.scatter(*Y.T)

# %% Test multiple implementations of numba
import time
import numpy as np	
from tallem.dimred import *
from tallem.distance import dist
from numba import prange
X = np.random.uniform(size=(500, 8), low=0.0, high=10.0)
D = dist(X, as_matrix=True)

tic = time.perf_counter()
for i in range(100):
	mds_cc = cmds(D, 10, method="numpy")
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
for i in range(100):
	mds_cc = cmds(D, 10, method="scipy")
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
for i in range(100):
	mds_cc = cmds_numba_naive(D, 10)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
for i in range(100):
	mds_cc = cmds_numba(D, 10)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
for i in range(100):
	mds_cc = landmark.cmds(D, 10)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

# %% Testing landmark MDS
tic = time.perf_counter()
for i in range(100):
	L_ind, L_rad = landmarks(X, 15)
	landmark_mds(X, L_ind, d=10)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

# %% Testing just landmark portion
L_ind, L_rad = landmarks(X, 15)
tic = time.perf_counter()
for i in range(100):
	landmark_mds(X, L_ind, d=10)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")


# %% Test MDS on many different subsets 
n = 155
grid_x, grid_y = np.meshgrid(range(n), range(n))
X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)	
subsets = [np.array(list(range(i*n, ((i+1)*n))), dtype=int) for i in range(n)]


# s_vec = np.ravel(subsets)
# s_ind = np.hstack((np.array([0]), np.cumsum([len(s) for s in subsets])))

# bench_parallel(X, s_vec, )



# %% 



def test_cmds_parallel_correct():
	import time
	import numpy as np
	from tallem.distance import dist
	from tallem.dimred import cmds
	from tallem.pbm import landmark

	## Verify can spawn threads 
	landmark.do_parallel(12)

	n = 155
	grid_x, grid_y = np.meshgrid(range(n), range(n))
	X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)	
	subsets = [np.array(list(range(i*n, ((i+1)*n))), dtype=int) for i in range(n)]
	
	## Run MDS sequentially
	mds_seq = [cmds(X[s,:], 2) for s in subsets]

	# Configure block sizes for the simple threading
	n_threads, J = 12, len(subsets)
	# blocks = np.asarray(list(range(0, J+1, int(J/n_threads))), dtype=int)
	blocks = np.append([0], np.cumsum(np.histogram(range(J), n_threads)[0]))
	assert len(blocks) == n_threads + 1
	blocks[n_threads] = J

	# X, cover_sets, d, n_threads, blocks
	# NOTE: this eats 'X'!
	print("beginning parallel MDS") 
	X_T = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=np.float64).T	
	mds_par = landmark.parallel_mds_blocks(X_T, subsets, 2, n_threads, blocks)
	max_error = np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))
	
	tic = time.perf_counter()
	mds_cc = [landmark.cmds(dist(X[s,:],as_matrix=True)**2, 2) for s in subsets]
	toc = time.perf_counter()
	print(f"{toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	mds_seq = [cmds(X[s,:], 2) for s in subsets]
	toc = time.perf_counter()
	print(f"{toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	mds_par = landmark.parallel_mds_blocks(X_T, subsets, 2, n_threads, blocks)
	toc = time.perf_counter()
	print(f"{toc - tic:0.4f} seconds")


	# landmark.cmds(dist(X, as_matrix=True), 2)
	# ## Verify correctness first
	# mds_par = landmark.parallel_cmds(X, subsets, 2, 1)
	# max_error = np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))
	# assert max_error <= np.finfo(np.float32).eps	
# 	# ## Verify correctness first
# 	# mds_par = landmark.parallel_cmds(X, subsets, 2, 12)
# 	# mds_seq = [cmds(X[s,:], 2) for s in subsets]
# 	# np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))
	# import time
	# tic = time.perf_counter()
	# landmark.parallel_cmds(X, subsets, 2, 12)
	# toc = time.perf_counter()
# 	# print(f"{toc - tic:0.4f} seconds")
# 	# tic = time.perf_counter()
# 	# mds_py = [cmds(X[s,:], 2) for s in subsets]
# 	# toc = time.perf_counter()
# 	# print(f"{toc - tic:0.4f} seconds")
# 	# # landmark.dist_matrix(X) == dist(X, as_matrix=True)**2
# 	# # %% 
# 	# import matplotlib.pyplot as plt
# 	# plt.scatter(*wut[0].T)
# 	# plt.scatter(*mds_py[0].T)
# 	# np.max(abs(dist(wut[0]) - dist(mds_py[0])))
# def test_isomap():
# 	import numpy as np
# 	from tallem.distance import dist
# 	from tallem.dimred import isomap, floyd_warshall, neighborhood_graph
# 	iso1 = isomap(X)
# 	Y = cmds(floyd_warshall(neighborhood_graph(X, k = 15).A))
# 	assert isinstance(Y, np.ndarray)
# def check_parallel_mds():
# 	import time
# 	from numba import jit, prange
# 	import numpy as np
# 	from tallem.dimred import cmds
# 	from tallem.pbm import landmark
# 	n = 250
# 	grid_x, grid_y = np.meshgrid(range(n), range(n))
# 	X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)	
# 	subsets = [np.array(list(range(i*n, ((i+1)*n))), dtype=int) for i in range(n)]
	
# 	@jit(nopython=False, parallel=True, forceobj=True)
# 	def parallel_mds(X, subsets):
# 		return([cmds(X[subsets[i],:]) for i in prange(len(subsets))])

# 	# %% 
# 	tic = time.perf_counter()
# 	s1 = [cmds(X[s,:]) for s in subsets]		
# 	toc = time.perf_counter()
# 	print(f"{toc - tic:0.4f} seconds")

# 	# %% 
# 	tic = time.perf_counter()
# 	s1 = [cmds(X[s,:], method="numpy") for s in subsets]		
# 	toc = time.perf_counter()
# 	print(f"{toc - tic:0.4f} seconds")

# 	# %% 
# 	tic = time.perf_counter()
# 	s2  = parallel_mds(X, subsets)
# 	toc = time.perf_counter()
# 	print(f"{toc - tic:0.4f} seconds")

# 	# %% 
# 	tic = time.perf_counter()
# 	landmark.parallel_cmds(X, subsets, 2, 1)
# 	toc = time.perf_counter()
# 	print(f"{toc - tic:0.4f} seconds")

# 	# %% 
# 	DM = [dist(X[subset,:], as_matrix=True) for subset in subsets]
# 	tic = time.perf_counter()
# 	mds_parallel_numba(DM)
# 	toc = time.perf_counter()
# 	print(f"{toc - tic:0.4f} seconds")
	

# #%% 
# @jit(nopython=True, parallel=False)
# def mds_numba(D, d: int = 2):
# 	''' MUST ACCEPT SQUARE DISTANCE MATRIX '''
# 	n = D.shape[0]
# 	H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
# 	B = -0.5 * H @ D @ H
# 	evals, evecs = np.linalg.eigh(B)
# 	evals, evecs = np.flip(evals), np.fliplr(evecs)
# 	evals, evecs = evals[range(d)], evecs[:,range(d)]
# 	w = np.where(evals > 0)[0]
# 	Y = np.zeros(shape=(n, d))
# 	Y[:,w] = evecs[:,w] @ np.diag(np.sqrt(evals[w]))
# 	return(Y)

# @jit(nopython=True, parallel=True)
# def mds_parallel_numba(dm, d: int = 2):
# 	results = [mds_numba(dm[i], d) for i in prange(len(dm))]
# 	return(results)
		

# from tallem.distance import dist
# n = 150
# grid_x, grid_y = np.meshgrid(range(n), range(n))
# X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)	
# subsets = [np.array(list(range(i*n, ((i+1)*n))), dtype=int) for i in range(n)]

import time
import numpy as np
from numba import njit, prange, config, threading_layer

x = np.random.uniform(size=500000)

config.THREADING_LAYER = 'tbb'
print(threading_layer())
# config.THREADING_LAYER = 'omp'

@njit
def ident_np(x):
	return np.cos(x) ** 2 + np.sin(x) ** 2

@njit
def ident_loops(x):
	r = np.empty_like(x)
	n = len(x)
	for i in range(n):
		r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
	return r

@njit(parallel=True, nogil=True)
def ident_parallel(x):
	return np.cos(x) ** 2 + np.sin(x) ** 2

@njit(parallel=True, nogil=True)
def ident_parallel2(x):
	n, r = len(x), np.empty_like(x)
	for i in prange(n):
		r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
	return r

tic = time.perf_counter()
y = ident_np(x)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
y = ident_loops(x)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
y = ident_parallel(x)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")

tic = time.perf_counter()
y = ident_parallel2(x)
toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")


import time
import numpy as np
from tallem.dimred import cmds
from tallem.distance import dist
from tallem.dimred import cmds_numba

for n in [100, 1e3, 2500, 5000, 1e4]:
	X = np.random.uniform(size=(int(n), 1000))
	D = dist(X, as_matrix=True)**2

	tic = time.perf_counter()
	y = cmds(D, d = 2, method="numpy")
	toc = time.perf_counter()
	print(f"np: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	y = cmds(D, d = 10, method="numpy")
	toc = time.perf_counter()
	print(f"np: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	y = cmds(D, d = 100, method="numpy")
	toc = time.perf_counter()
	print(f"np: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	y = cmds(D, d = 2, method="scipy")
	toc = time.perf_counter()
	print(f"sp: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	y = cmds(D, d = 10, method="scipy")
	toc = time.perf_counter()
	print(f"sp: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	y = cmds(D, d = 100, method="scipy")
	toc = time.perf_counter()
	print(f"sp: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	y = cmds_numba(D, 100) # conclusion: faster than numpy, slower than scipy, due to estimating all eiegnvectors
	toc = time.perf_counter()
	print(f"numba: {toc - tic:0.4f} seconds")


import time
import numpy as np
from tallem.dimred import cmds, cmds_numba, landmark_cmds_numba
from tallem.distance import dist


D = dist(X, as_matrix=True)**2
np.max(abs(dist(cmds_numba(D, 2)) - dist(cmds(D, d=2))))


from tallem.samplers import landmarks

X = np.random.uniform(size=(100, 2))
X = np.hstack((X, np.random.uniform(size=X.shape, low=0.0, high=0.15)))

Lind, Lrad = landmarks(X, k  = X.shape[0])

dx = dist(cmds(X, d=2))
for i in range(3, X.shape[0]):
	ind = np.take(Lind, range(i))
	PD = dist(X[ind,:], X)
	LD = PD[:,ind]
	max_error = np.max(abs(dist(landmark_cmds_numba(LD**2, PD**2, d=2)) - dx))
	print(max_error)


