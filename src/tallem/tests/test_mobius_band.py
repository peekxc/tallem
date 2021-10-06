# test_mobius_band.py 
print(f'Running: {__name__}')

import unittest
import numpy as np
from tallem import TALLEM
from tallem.datasets import mobius_band
from tallem.cover import IntervalCover

class TestMobiusBand():
	def test_data_set(self):
		M = mobius_band(plot=False, embed=6)
		X, B = M['points'], M['parameters'][:,[1]]
		assert isinstance(X, np.ndarray)
		assert isinstance(B, np.ndarray)

	def test_embedding(self):
		M = mobius_band(plot=False, embed=6)
		X, B = M['points'], M['parameters'][:,[1]]
		m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
		cover = IntervalCover(B, n_sets = 10, overlap = 0.20, space = [0, 2*np.pi], metric = m_dist)
		embedding = TALLEM(cover, local_map="pca2", n_components=3).fit_transform(X, B)
		assert isinstance(embedding, np.ndarray)