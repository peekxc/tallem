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



def test_mds_cython():
	from tallem.extensions import mds_cython