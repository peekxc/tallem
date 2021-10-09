# test_cover.py 
import unittest
import sys 

print(sys.path)

# from tallem.pbm import landmark

from pytest import approx
import numpy as np	


## Test landmarks
from tallem.samplers import landmarks
def test_landmarks():
	x = np.random.uniform(size=(100,2))
	ind, radii = landmarks(x, k = 15)
	assert len(ind) == 15
	assert len(radii) == 15
	assert np.all(np.argsort(radii) == np.array(range(15))[::-1])

from tallem.cover import validate_cover
from tallem.cover import IntervalCover
def test_interval_cover_1d():	
	x = np.random.uniform(size=(100,1), low = 0.0, high = 1.0)
	cover = IntervalCover(x, n_sets = 10, overlap = 0.20)
	assert validate_cover(x.shape[0], cover)

def test_interval_cover_2d():	
	x = np.random.uniform(size=(100,2), low = 0.0, high = 1.0)
	cover = IntervalCover(x, n_sets = 10, overlap = 0.20)
	assert validate_cover(x.shape[0], cover)
