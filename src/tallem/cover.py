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


from enum import IntEnum
class Gluing(IntEnum):
	INVERTED = -1 
	NONE = 0
	ALIGNED = 1

## TODO: somehow extend this to allow identification of edges
## maybe: identify = [i_0, ..., i_d] where i_k \in [-1, 0, +1], where -1 indicates reverse orientation, 0 no identification, and +1 regular orientation
class IntervalCover():
	'''
	A Cover *is* an iterable
	Cover -> divides data into subsets, satisfying covering property, retains underlying neighbor structures + metric (if any) 
	parameters: 
		n_sets := number of cover elements to use to create the cover 
		method := method to create the cover. Can be 'landmark', in which case landmarks are used to define the subsets, or 
		optionally 'grid', in which case a uniformly-spaced grid is imposed over the data. 
	'''
	def __init__(self, a: npt.ArrayLike, n_sets: List[int], overlap: List[float], gluing: Optional[List[Gluing]] = None, implicit: bool = False, **kwargs):
		self.n_points, self.dimension = a.shape[0], a.shape[1]
		self.metric = "euclidean"
		self.implicit = implicit
		self.n_sets = np.repeat(n_sets, self.dimension) if (isinstance(n_sets, int) or len(n_sets) == 1) else np.array(n_sets)
		self.overlap = np.repeat(overlap, self.dimension) if (isinstance(overlap, float) or len(overlap) == 1) else np.array(overlap)
		self.bbox = np.vstack((np.amin(a, axis=0), np.amax(a, axis=0)))
		self.base_width = np.diff(self.bbox, axis = 0)/self.n_sets
		self.set_width = self.base_width + (self.base_width*self.overlap)/(1.0 - self.overlap)
			
		self.gluing = np.repeat(Gluing.NONE, self.dimension) if gluing is None else gluing
		if len(self.gluing) != self.dimension:
			raise ValueError("Invalid gluing vector given. Must match the dimension of the cover.")
		
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
			self.n, self.D = a.shape[0], a.shape[1]
			if self.implicit:
				self._data = a
			else: 
				self.sets = self.construct(a)
		return self

	def _diff_to(self, x: npt.ArrayLike, point: npt.ArrayLike):
		i_dist = np.zeros(x.shape[0])
		for d_i in range(self.dimension):
			diff = abs(x[:,d_i] - point[d_i])
			if self.gluing[d_i] == 0:
				i_dist[:,d_i] += diff
			elif self.gluing[d_i] == 1:
				rng = abs(self.bbox[1,d_i] - self.bbox(0,d_i))
				i_dist += np.minimum(diff, abs(rng - diff))
			elif self.gluing[d_i] == -1:
				rng = abs(self.bbox[1,d_i] - self.bbox(0,d_i))
				coords = self.bbox[1,d_i] - x[:,d_i] # Assume inside the box
				diff = abs(coords - point[d_i])
				i_dist += np.minimum(diff, abs(rng - coords))

			# self.bbox[0,:]
			# np.minimum(1.0 - abs(x[:,0] - centroid[0]), abs(x[:,0] - centroid[0]))

	def __len__(self):
		return(np.prod(self.n_sets))

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

## A Partition oif unity is 
## B := point cloud topologiucal space
## phi := function mapping a subset of m points to (m x J) matrix 
def partition_of_unity(B: npt.ArrayLike, cover: Iterable, beta: Union[str, Callable[npt.ArrayLike, npt.ArrayLike]] = "triangular", weights: Optional[npt.ArrayLike] = None):
	J = len(cover)
	weights = np.ones(J) if weights is None else np.array(weights)
	if len(weights) != J:
		raise ValueError("weights must have length matching the number of sets in the cover.")
	if beta is None:
		raise ValueError("phi map must be a real-valued function, or a string indicating one of the precomputed ones.")
	
	## Derive centroids, use dB metric to define distances => partition of unity to each subset 
	if isinstance(cover, IntervalCover):
		max_r = np.linalg.norm(cover.set_width)
		def beta(cover_set):
			index, subset = cover_set
			centroid = cover.bbox[0,:] + (np.array(index) * cover.base_width) + cover.base_width/2.0
			beta_j = np.maximum(max_r - dist(B, centroid), 0.0)
			beta_j[subset] = 0.0
			return(beta_j)
	else: 
		raise ValueError("Only interval cover is supported")
			
			# cover.in dist(B, centroid)
	# if phi is not None: 
	# 	if phi == "triangular":
	# 	elif phi == "":

	# Apply the phi map to each subset, collecting the results into lists
	row_indices, beta_image = [], []
	for index, subset in cover: 
		
		## varphi_j represents the (beta_j \circ f)(X) = \beta_j(B)
		## As such, the vector returned should be of length n
		varphi_j = beta((index, subset))
		if len(varphi_j) != cover.n: raise ValueError("Alignment function 'beta' must return a set of values for every point in X.")

		## Record non-zero row indices + the function values themselves
		row_indices.append(np.nonzero(varphi_j)[0])
		beta_image.append(varphi_j[row_indices[-1]])

	## Use a CSC-sparse matrix to represent the partition of unity
	row_ind = np.hstack(row_indices)
	col_ind = np.repeat(range(J), [len(subset) for subset in row_indices])
	pou = csc_matrix((np.hstack(phi_image), (row_ind, col_ind)))
	pou /= np.sum(pou, axis = 1)

	## The final partition of unity weights elements in the cover over B
	return(pou)
	

def partition_of_unity(a: npt.ArrayLike, centers: npt.ArrayLike, radius: np.float64, d = dist) -> csc_matrix:
	'''
	Partitions 'a' into a partition of unity using a tent function. 
	If m points are partitioned by n center points, then 
	the result is a (m x n) matrix of weights yielding the normalized 
	distance from each point to the given set of centers. Each row 
	is normalized to sum to 1. 
	'''
	a = np.array(a)
	centers = np.array(centers)
	P = np.array(np.maximum(0, radius - d(a, centers)), dtype = np.float32)
	P = (P.T / np.sum(P, axis = 1)).T
	return(P)