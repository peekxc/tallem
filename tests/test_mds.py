# test_mobius_band.py 
print(f'Running: {__name__}')

def test_classical_mds():
	import numpy as np
	from tallem.distance import dist
	from tallem.dimred import cmds
	n = 25
	grid_x, grid_y = np.meshgrid(range(n), range(n))
	X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)

	mds1 = cmds(X)
	mds2 = cmds(dist(X)**2)
	mds3 = cmds(dist(X, as_matrix=True)**2)
	assert isinstance(mds1, np.ndarray)
	assert isinstance(mds2, np.ndarray)
	assert np.max(abs(mds1 - mds2)) <= np.finfo(np.float32).eps
	assert np.max(abs(mds2 - mds3)) <= np.finfo(np.float32).eps

def test_classical_mds_cpp():
	import numpy as np
	from tallem.distance import dist
	from tallem.dimred import cmds
	from tallem.pbm import landmark
	n = 25
	grid_x, grid_y = np.meshgrid(range(n), range(n))
	X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)
	
	assert True
	# X.shape[0]
	# evalues, evecs = cmds(X, 2, coords=False)
	# Y = landmark.cmds(dist(X, as_matrix=True)**2, 2)

	# import matplotlib.pyplot as plt
	# plt.scatter(*evecs.T)
	# plt.scatter(*Y[:,0:2].T)
	# plt.scatter(*cmds(X).T)

# def test_dist_matrix():
# 	np.all(landmark.dist_matrix(X) == dist(X, as_matrix=True))
# def test_classical_mds_parallel():
# 	import numpy as np
# 	from tallem.distance import dist
# 	from tallem.dimred import cmds
# 	from tallem.pbm import landmark
# 	n = 150
# 	grid_x, grid_y = np.meshgrid(range(n), range(n))
# 	X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)	
# 	subsets = [np.array(list(range(i*n, ((i+1)*n))), dtype=int) for i in range(n)]
	
# 	assert True
# 	# ## Verify correctness first
# 	# mds_par = landmark.parallel_cmds(X, subsets, 2, 1)
# 	# mds_seq = [cmds(X[s,:], 2) for s in subsets]
# 	# np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))

# 	# ## Verify correctness first
# 	# mds_par = landmark.parallel_cmds(X, subsets, 2, 12)
# 	# mds_seq = [cmds(X[s,:], 2) for s in subsets]
# 	# np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))
# 	# import time
# 	# tic = time.perf_counter()
# 	# landmark.parallel_cmds(X, subsets, 2, 12)
# 	# toc = time.perf_counter()
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


# DM = [dist(X[subset,:], as_matrix=True) for subset in subsets]

