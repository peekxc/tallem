# test_cover.py 
import unittest
import sys 
from pytest import approx

print(sys.path)

def test_landmarks():
	import numpy as np
	from tallem.samplers import landmarks
	x = np.random.uniform(size=(100,2))
	ind, radii = landmarks(x, k = 15)
	assert len(ind) == 15
	assert len(radii) == 15
	assert np.all(np.argsort(radii) == np.array(range(15))[::-1])

def test_interval_cover_1d():	
	import numpy as np
	from tallem.cover import validate_cover
	from tallem.cover import IntervalCover
	x = np.random.uniform(size=(100,1), low = 0.0, high = 1.0)
	cover = IntervalCover(x, n_sets = 10, scale = 1.20)
	assert validate_cover(x.shape[0], cover)

	index_set = list(cover.keys())
	for index in index_set:
		d = cover.bump(x, index)
		assert np.all(cover[index] == np.flatnonzero(d >= 0))

	for index in index_set:
		d = cover.bump(x, index, normalize=True)
		assert np.all(cover[index] == np.flatnonzero(d >= 0))
		assert np.all(d <= 1.0)

def test_interval_cover_2d():	
	import numpy as np
	from tallem.cover import validate_cover
	from tallem.cover import IntervalCover
	
	x = np.random.uniform(size=(100,2), low = 0.0, high = 1.0)
	cover = IntervalCover(x, n_sets = (5, 5), scale = (1.20, 1.20))
	assert validate_cover(x.shape[0], cover)

	## Test set distance method
	index_set = list(cover.keys())
	for index in index_set:
		d = cover.bump(x, index)
		assert np.all(cover[index] == np.flatnonzero(d >= 0))

	for index in index_set:
		d = cover.bump(x, index, normalize=True)
		assert np.all(cover[index] == np.flatnonzero(d >= 0))
		assert np.all(d <= 1.0)

def test_landmark_cover():
	import numpy as np
	from tallem.cover import LandmarkCover, validate_cover
	x = np.random.uniform(size=(100,2), low = 0.0, high = 1.0)
	cover = LandmarkCover(x, n_sets = 15)
	assert validate_cover(x.shape[0], cover)

	from tallem.distance import dist
	cover2 = LandmarkCover(dist(x), n_sets = 15)
	assert validate_cover(x.shape[0], cover2)

	cover3 = LandmarkCover(dist(x, as_matrix=False), n_sets = 15)
	assert validate_cover(x.shape[0], cover3)

	# Ensure point-based Landmark covers support bump functions for index-like inputs
	index_set = list(cover.keys())
	for index in index_set:
		bf = cover.bump(cover[index], index) 
		assert isinstance(bf, np.ndarray)
		assert np.all(bf >= 0.0)

	# Ensure point-based Landmark covers support bump functions for point-like inputs
	for index in index_set:
		ind1 = np.flatnonzero(cover.bump(x, index) > 0.0)
		ind2 = np.flatnonzero(cover.bump(x, index) >= 0.0)
		if len(cover[index]) == len(ind1):
			assert np.all(cover[index] == ind1)
		elif (len(cover[index]) == len(ind2)):
			assert np.all(cover[index] == ind2)
		else: 
			assert False
	
	# Ensure distance-based Landmark covers support bump functions for index-like inputs
	for index in index_set:
		bf = cover2.bump(cover2[index], index) 
		assert isinstance(bf, np.ndarray)
		assert np.all(bf >= 0.0)

def test_circle_cover():
	import numpy as np
	from tallem.cover import CircleCover, validate_cover
	x = np.random.uniform(size=(100,), low = 0.0, high = 2*np.pi)
	cover = CircleCover(x, n_sets = 10, scale = 1.5)
	assert validate_cover(x.shape[0], cover)

	index_set = list(cover.keys())
	for index in index_set:
		d = cover.bump(x, index)
		assert np.all(cover[index] == np.flatnonzero(d >= 0))

	for index in index_set:
		d = cover.bump(x, index, normalize=True)
		assert np.all(cover[index] == np.flatnonzero(d >= 0))
		assert np.all(d <= 1.0)

def check_neighborhood_graphs():
	import numpy as np
	import tallem.dimred 
	x = np.random.uniform(size=(100,2), low = 0.0, high = 1.0)

	from sklearn.neighbors import kneighbors_graph
	g1 = kneighbors_graph(x, n_neighbors=15, mode='distance').tocsc()
	g2 = knn_graph(x, k = 15).tocsc()
	max_diff = np.max(abs(g1.A - g2.A))

	from sklearn.neighbors import radius_neighbors_graph
	g1 = radius_neighbors_graph(x, radius=0.15, mode='distance').tocsc()
	g2 = rnn_graph(x, r = 0.15).tocsc()
	max_diff = np.max(abs(g1.A - g2.A))





