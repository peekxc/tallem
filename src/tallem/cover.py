## Plan: 
## Cover -> divides data into subsets, satisfying covering property, retains underlyign neighbor structures + metric (if any)  
## Local euclidean model -> just transforms each subset from the cover ; separate from cover and PoU! 
## PoU -> Takes a cover as input, a set of weights for each point, and a function yielding a set of real values indicating 
## how "close" each point is to each cover subset; if not supplied, this is inferred from the metric, otherwise a tent function 
## or something is used. Ultimately returns an (n x J) sparse matrix where each row is normalized to sum to 1. 
import numpy as np
import numpy.typing as npt
from sklearn.neighbors import BallTree
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from tallem.distance import dist 

# GridCover ?
# LandmarkCover ? 
# Bitvector vs sparsematrix storage option ?
# Query: implicit vs explicit

from typing import Callable, Iterable, List, Set, Dict, Optional, Tuple, Any, Union, Sequence
from itertools import combinations, product

class IntervalCover():
	'''
	A Cover *is* an iterable
	Cover -> divides data into subsets, satisfying covering property, retains underlying neighbor structures + metric (if any) 
	parameters: 
		n_sets := number of cover elements to use to create the cover 
		method := method to create the cover. Can be 'landmark', in which case landmarks are used to define the subsets, or 
		optionally 'grid', in which case a uniformly-spaced grid is imposed over the data. 
	'''
	def __init__(self, a: npt.ArrayLike, n_sets: List[int], overlap: List[float], implicit: bool = False, **kwargs):
		D = a.shape[1]
		self.implicit = implicit		
		self.n_sets = np.repeat(n_sets, D) if (isinstance(n_sets, int) or len(n_sets) == 1) else np.array(n_sets)
		self.overlap = np.repeat(overlap, D) if (isinstance(overlap, float) or len(overlap) == 1) else np.array(overlap)
		self.bbox = np.vstack((np.amin(a, axis=0), np.amax(a, axis=0)))
		self.base_width = np.diff(self.bbox, axis = 0)/self.n_sets
		self.set_width = self.base_width + (self.base_width*self.overlap)/(1.0 - self.overlap)
			
		## If implicit = False, then construct the sets and store them
		## otherwise, they must be constructed on demand via __call__
		if not(self.implicit): 
			self.sets = self.construct(a)

	def construct(self, a: npt.ArrayLike, index: Optional[npt.ArrayLike] = None):
		if index:
			centroid = self.bbox[0,:] + (np.array(index) * self.base_width) + self.base_width/2.0
			return(np.ravel(np.where(np.bitwise_and.reduce(abs(a - centroid) <= self.set_width/2.0, axis = 1))))
		else:
			cover_sets = { }
			for index in np.ndindex(*self.n_sets):
				centroid = self.bbox[0,:] + (np.array(index) * self.base_width) + self.base_width/2.0
				cover_sets[index] = np.ravel(np.where(np.bitwise_and.reduce(abs(a - centroid) <= self.set_width/2.0, axis = 1)))
			return(cover_sets)
	
	def __iter__(self):
		if self.implicit:
			for index in np.ndindex(*self.n_sets):
				centroid = self.bbox[0,:] + (np.array(index) * self.base_width) + self.base_width/2.0
				point_idx = np.ravel(np.where(np.bitwise_and.reduce(abs(self._data - centroid) <= self.set_width/2.0, axis = 1)))
				yield np.array(index), point_idx 
			self._data = None
		else: 
			for index, cover_set in self.sets.items():
				yield index, cover_set

	def __call__(self, a: Optional[npt.ArrayLike]):
		if a is not None:
			if self.implicit:
				self._data = a
			else: 
				self.sets = self.construct(a)
		return self

	def __repr__(self) -> str:
		return("Interval Cover")




## TODO: Use code below to partition data set, then use multi-D "floodfill" type search 
## to obtain O(n + k) complexity. Could also bound how many sets overlap and search all k of those
## to yield O(nk) vectorized
## breaks = [bin_width[d]*np.array(list(range(n_sets[d]))) for d in range(D)]
## np.reshape([np.digitize(x[:,d], breaks[d]) for d in range(D)], x.shape)

# prop_overlap <- self$percent_overlap/100
# base_interval_length <- filter_len/self$number_intervals
# interval_length <- base_interval_length + (base_interval_length * prop_overlap)/(1.0 - prop_overlap)
# eps <- (interval_length/2.0) + sqrt(.Machine$double.eps) ## ensures each point is in the cover
# set_bounds <- do.call(rbind, lapply(index, function(idx_str){
#   idx <- strsplit(substr(idx_str, start=2L, stop=nchar(idx_str)-1L), split = " ")[[1]]
#   centroid <- filter_min + ((as.integer(idx)-1L)*base_interval_length) + base_interval_length/2.0
#   return(c(centroid - eps, centroid + eps))
# }))



# def LandmarkCover:
# 	def __init__(self):
# 			self.neighbor = BallTree(a, kwargs)

def partition_of_unity(a: npt.ArrayLike, centers: npt.ArrayLike, radius: np.float64, d = dist) -> csc_matrix:
	'''
	Partitions 'a' into a partition of unity using a tent function. 
	If m points are partitioned by n center points, then 
	the result is a (m x n) matrix of weights yielding the normalized 
	distance from each point to the given set of centers. Each row 
	is normalized to sum to 1. 
	'''
	if isinstance(a, IntervalCover):
		print("")
	a = np.array(a)
	centers = np.array(centers)
	P = np.array(np.maximum(0, radius - d(a, centers)), dtype = np.float32)
	P = (P.T / np.sum(P, axis = 1)).T
	return(P)