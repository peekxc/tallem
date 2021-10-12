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

def test_classical_mds_parallel():
	import numpy as np
	from tallem.distance import dist
	from tallem.dimred import cmds
	from tallem.pbm import landmark
	n = 150
	grid_x, grid_y = np.meshgrid(range(n), range(n))
	X = np.array(np.c_[grid_x.flatten(), grid_y.flatten()], dtype=float)	
	subsets = [np.array(list(range(i*n, ((i+1)*n))), dtype=int) for i in range(n)]
	
	assert True
	# ## Verify correctness first
	# mds_par = landmark.parallel_cmds(X, subsets, 2, 1)
	# mds_seq = [cmds(X[s,:], 2) for s in subsets]
	# np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))

	# ## Verify correctness first
	# mds_par = landmark.parallel_cmds(X, subsets, 2, 12)
	# mds_seq = [cmds(X[s,:], 2) for s in subsets]
	# np.max(np.array([max(abs(dist(p) - dist(q))) for p,q in zip(mds_par, mds_seq)]))

	# import time
	# tic = time.perf_counter()
	# landmark.parallel_cmds(X, subsets, 2, 12)
	# toc = time.perf_counter()
	# print(f"{toc - tic:0.4f} seconds")

	# tic = time.perf_counter()
	# mds_py = [cmds(X[s,:], 2) for s in subsets]
	# toc = time.perf_counter()
	# print(f"{toc - tic:0.4f} seconds")
	# # landmark.dist_matrix(X) == dist(X, as_matrix=True)**2
	
	# # %% 
	# import matplotlib.pyplot as plt
	# plt.scatter(*wut[0].T)
	# plt.scatter(*mds_py[0].T)
	# np.max(abs(dist(wut[0]) - dist(mds_py[0])))

		

def test_isomap():
	import numpy as np
	from tallem.distance import dist
	from tallem.dimred import isomap, floyd_warshall, neighborhood_graph
	iso1 = isomap(X)
	Y = cmds(floyd_warshall(neighborhood_graph(X, k = 15).A))
	assert isinstance(Y, np.ndarray)


	## Python version 
	# %%
	%time 

