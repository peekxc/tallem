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

	X = np.random.uniform(size=(100,3))
	mds1 = cmds(X, 3, method="fortran")
	mds2 = cmds(X, 3, method="numpy")
	mds3 = cmds(X, 3, method="scipy")
	assert np.max(abs(dist(mds1) - dist(mds1))) <= np.finfo(np.float32).eps
	assert np.max(abs(dist(mds2) - dist(mds3))) <= np.finfo(np.float32).eps


def test_pca():
	import numpy as np
	from tallem.distance import dist
	from tallem.dimred import cmds, pca
	X = np.random.uniform(size=(100,3))
	Y = pca(X)
	assert isinstance(Y, np.ndarray)
	error = abs(dist(Y) - dist(cmds(X)))
	assert np.max(error) <= np.finfo(np.float32).eps



# def test_mds_cython():
# 	import numpy as np
# 	from tallem.extensions import mds_cython

# 	## Prepare data + subsets 
# 	n_sets = 350
# 	X = np.asfortranarray(np.random.uniform(size=(5000,3)), dtype=np.float64)
# 	subsets = [np.random.choice(X.shape[0], size=X.shape[0]//n_sets) for i in range(n_sets)]

# 	## Preprocessing 
# 	ind_vec, len_vec = mds_cython.flatten_list_of_lists(subsets)
# 	max_n =  np.max([len(s) for s in subsets])
# 	sum_n = np.sum([len(s) for s in subsets])
# 	results = np.zeros((2, sum_n), dtype=np.float64, order='F') ## Output 

# 	mds_cython.cython_cmds_parallel(X, 2, ind_vec, len_vec, max_n, results)
# 	np.sum(results != 0.0)

# 	models = np.hsplit(results, len_vec[1:(len(len_vec)-1)])

# 	from tallem.dimred import cmds
# 	from tallem.distance import dist
# 	true_dist = dist(cmds(X[subsets[0], :], 2))
# 	test_dist = dist(models[0].T)
# 	np.max(abs(true_dist - test_dist))

# 	import timeit
# 	res = timeit.timeit(lambda: mds_cython.cython_cmds_parallel(X, 2, ind_vec, len_vec, max_n, results), number = 50)
# 	print(f"{res:0.4f} seconds")

# 	res = timeit.timeit(lambda: [cmds(X[s,:], 2) for s in subsets], number = 50)
# 	print(f"{res:0.4f} seconds")

# 	import line_profiler
# 	profile = line_profiler.LineProfiler()
# 	profile.add_function(mds_cython.cython_cmds_parallel)
# 	profile.add_function(mds_cython.dist_matrix_subset)
# 	profile.add_function(mds_cython.cython_cmds_fortran_inplace)
# 	profile.add_function(mds_cython.cython_dsyevr_inplace)
# 	profile.enable_by_count()
# 	mds_cython.cython_cmds_parallel(X, 2, ind_vec, len_vec, max_n, results)
# 	profile.print_stats(output_unit=1e-3)

