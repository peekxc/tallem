# test_mobius_band.py 
print(f'Running: {__name__}')

import unittest
import numpy as np
from tallem import TALLEM
from tallem.datasets import mobius_band
from tallem.cover import IntervalCover, CoverLike, validate_cover

def test_mobius_data_set():
	M_extrinsic, M_intrinsic = mobius_band()
	assert isinstance(M_extrinsic, np.ndarray)
	assert isinstance(M_intrinsic, np.ndarray)

def test_embedding():
	X, B = mobius_band()
	polar_coordinate = B[:,[1]]

	m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
	cover = IntervalCover(polar_coordinate, n_sets = 10, overlap = 0.20, space = [0, 2*np.pi], metric = m_dist)

	## Ensure the cover is a valid cover
	assert isinstance(cover, IntervalCover)
	assert isinstance(cover, CoverLike)
	assert validate_cover(X.shape[0], cover)

	## Test TALLEM constructor works
	top = TALLEM(cover, local_map="pca2", n_components=3)
	assert top.__class__ == TALLEM

	## Ensure the whole pipeline works
	embedding = top.fit_transform(X, polar_coordinate)
	assert isinstance(embedding, np.ndarray)
	assert embedding.shape[0] == X.shape[0]
	assert embedding.shape[1] == 3
