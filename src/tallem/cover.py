## Plan: 
## Cover -> divides data into subsets, satisfying covering property, retains underlyign neighbor structures + metric (if any)  
## Local euclidean model -> just transforms each subset from the cover ; separate from cover and PoU! 
## PoU -> Takes a cover as input, a set of weights for each point, and a function yielding a set of real values indicating 
## how "close" each point is to each cover subset; if not supplied, this is inferred from the metric, otherwise a tent function 
## or something is used. Ultimately returns an (n x J) sparse matrix where each row is normalized to sum to 1. 
import numpy as np
import numpy.typing as npt
from sklearn.neighbors import BallTree
from scipy.sparse import csc_matrix, diags
from scipy.sparse.csgraph import minimum_spanning_tree,connected_components 
from .distance import dist 
from .utility import find_where

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
		space := the type of space to construct a cover on. Can be a point set, a set of intervals (per column) indicating the domain of the cover, 
						 or a string indicating a common configuration space. 
		n_sets := number of cover elements to use to create the cover 
		overlap := proportion of overlap between adjacent sets per dimension, as a number in [0,1). It is recommended, but not required, to keep this parameter below 0.5. 
		gluing := optional, whether to identify the sides of the cover as aligned in the same direction (1), in inverted directions (-1), or no identification (0) (default). 
		implicit := optional, whether to store all the subsets in a precomputed dictionary, or construct them on-demand. Defaults to the former.
	'''
	def __init__(self, space: Union[npt.ArrayLike, str], n_sets: List[int], overlap: List[float], gluing: Optional[List[Gluing]] = None, implicit: bool = False, **kwargs):
		
		## Detect if one is creating the cover an a specified configuration space, or if the boundary box dimensions were given, etc. 
		if isinstance(space, str):
			raise ValueError("Specifying the configuration space by name hasn't been implemented yet.")
		else: 
			space = np.array(space, copy=False)
			if len(space.shape) == 1: space = np.reshape(space, (len(space), 1))
			if len(space.shape) != 2: raise ValueError("Invalid data shape. Must be a matrix.")
			
			## Two cases: 
			## 	1) if only two rows, assume the i'th column gives the range of the i'th dimension 
			## 	2) otherwise, infer the boundaries of the domain from the data directly 
			self.dimension = space.shape[1]
			if space.shape[0] == 2: 
				self.bbox = space
				self._data = None
			else: 
				self.bbox = np.vstack((np.amin(space, axis=0), np.amax(space, axis=0)))
				self._data = np.array(space, copy=False)

		## Set intrinsic properties of the cover
		self.metric = "euclidean"
		self.implicit = implicit
		self.n_sets = np.repeat(n_sets, self.dimension) if (isinstance(n_sets, int) or len(n_sets) == 1) else np.array(n_sets)
		self.overlap = np.repeat(overlap, self.dimension) if (isinstance(overlap, float) or len(overlap) == 1) else np.array(overlap)
		
		## Assert these parameters match the dimension of the cover prior to assignment
		if len(self.n_sets) != self.dimension or len(self.overlap) != self.dimension:
			raise ValueError("Dimensions mismatch: supplied set or overlap arity does not match dimension of the cover.")
		self.base_width = np.diff(self.bbox, axis = 0)/self.n_sets
		self.set_width = self.base_width + (self.base_width*self.overlap)/(1.0 - self.overlap)
		
		## Choose how to handle edge orientations
		self.gluing = np.repeat(Gluing.NONE, self.dimension) if gluing is None else gluing
		if len(self.gluing) != self.dimension:
			raise ValueError("Invalid gluing vector given. Must match the dimension of the cover.")
		
		## If implicit = False, then construct the sets and store them
		## otherwise, they must be constructed on demand via __call__
		if not(self.implicit) and not(self._data is None): 
			self.sets = self.construct(self._data)

	def _diff_to(self, x: npt.ArrayLike, point: npt.ArrayLike):
		i_dist = np.zeros(x.shape)
		for d_i in range(self.dimension):
			diff = abs(x[:,d_i] - point[:,d_i])
			if self.gluing[d_i] == 0:
				i_dist[:,d_i] = diff
			elif self.gluing[d_i] == 1:
				rng = abs(self.bbox[1:2,d_i] - self.bbox[0:1,d_i])
				i_dist[:,d_i] = np.minimum(diff, abs(rng - diff))
			elif self.gluing[d_i] == -1:
				rng = abs(self.bbox[1:2,d_i] - self.bbox[0:1,d_i])
				coords = self.bbox[1:2,d_i] - x[:,d_i] # Assume inside the box
				diff = abs(coords - point[d_i])
				i_dist[:,d_i] = np.minimum(diff, abs(rng - coords))
			else:
				raise ValueError("Invalid value for gluing parameter")
		return(i_dist)
	
	def construct(self, a: npt.ArrayLike, index: Optional[npt.ArrayLike] = None):
		if index is not None:
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + self.base_width/2.0
			diff = self._diff_to(a, centroid)
			return(np.ravel(np.where(np.bitwise_and.reduce(diff <= self.set_width/2.0, axis = 1))))
		else:
			cover_sets = { index : self.construct(a, index) for index in np.ndindex(*self.n_sets) }
			return(cover_sets)
	
	def __iter__(self):
		if self.implicit:
			for index in np.ndindex(*self.n_sets):
				subset = construct(self._data, index)
				yield np.array(index), subset
		else: 
			for index, cover_set in self.sets.items():
				yield index, cover_set

	def __getitem__(self, index):
		if isinstance(index, tuple):
			return(self.construct(self._data, index) if self.implicit else self.sets[index])
		elif int(index) == index:
			index = list(self.sets.keys())[index]
			return(index, self.construct(self._data, index) if self.implicit else self.sets[index])
		else: 
			raise ValueError("Cover must be indexed by either a tuple or an integer index")
		
	def __call__(self, a: Optional[npt.ArrayLike]):
		if a is not None:
			a = np.array(a, copy=False)
			if a.shape[2] != self.dimension: raise ValueError("The dimensionality of the supplied data does not match the dimension of the cover.")
			self._data = a
			if not self.implicit:
				self.sets = self.construct(a)
		return self
	
	def __len__(self):
		return(np.prod(self.n_sets))

	def __repr__(self) -> str:
		return("Interval Cover")
	
	def plot(self):
		if self.dimension == 1:
			import matplotlib.pyplot as plt
			import matplotlib.patches as patches
			fig, ax = plt.subplots(figsize=(12, 6))
			rng = np.abs(self.bbox[1] - self.bbox[0])
			plt.xlim([self.bbox[0]-(0.15*rng), self.bbox[1]+(0.15*rng)])
			plt.ylim([-1, 1])
			plt.hlines(0,self.bbox[0],self.bbox[1], colors="black")
			frame = plt.gca()
			frame.axes.get_yaxis().set_visible(False)
			for index, _ in self:
				centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + self.base_width/2.0 
				anchor = (centroid - self.set_width/2.0, -0.5)
				rect = patches.Rectangle(anchor, self.set_width, 1.0, linewidth=0.5, edgecolor='black', facecolor='#BEBEBE33')
				ax.add_patch(rect)
			plt.eventplot(np.ravel(self._data), orientation = "horizontal", linewidths=0.05, lineoffsets=0)


	@property
	def data(self):
		return(self._data)

	@property
	def index_set(self):
		return(list(np.ndindex(*self.n_sets) if self.implicit else self.sets.keys()))

	def validate(self):
		elements = { subset for index, subset in cover }
		return(len(elements) >= self.n)	

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
def partition_of_unity(B: npt.ArrayLike, cover: Iterable, beta: Union[str, Callable[npt.ArrayLike, npt.ArrayLike]] = "triangular", weights: Optional[npt.ArrayLike] = None) -> csc_matrix:
	if (B.ndim != 2): raise ValueError("Error: filter must be matrix.")
	J = len(cover)
	weights = np.ones(J) if weights is None else np.array(weights)
	if len(weights) != J:
		raise ValueError("weights must have length matching the number of sets in the cover.")
	if beta is None:
		raise ValueError("phi map must be a real-valued function, or a string indicating one of the precomputed ones.")
	
	## Derive centroids, use dB metric to define distances => partition of unity to each subset 
	if isinstance(cover, IntervalCover):
		max_r = np.linalg.norm(cover.set_width)/2.0
		def beta(cover_set):
			index, subset = cover_set
			centroid = cover.bbox[0:1,:] + (np.array(index) * cover.base_width) + cover.base_width/2.0
			# beta_j = np.maximum(max_r - dist(B, centroid), 0.0) ## use triangular
			dist_to_poles = np.sqrt(np.sum(cover._diff_to(B, centroid)**2, axis = 1))
			beta_j = np.maximum(max_r - dist_to_poles, 0.0)
			## TODO: rework so this isn't needed!
			beta_j[np.setdiff1d(range(B.shape[0]), subset)] = 0.0
			return(beta_j)
	else: 
		raise ValueError("Only interval cover is supported for now.")

	# Apply the phi map to each subset, collecting the results into lists
	row_indices, beta_image = [], []
	for index, subset in cover: 
		
		## varphi_j represents the (beta_j \circ f)(X) = \beta_j(B)
		## As such, the vector returned should be of length n
		varphi_j = beta((index, subset))
		# if len(varphi_j) != cover.n_points: raise ValueError("Alignment function 'beta' must return a set of values for every point in X.")

		## Record non-zero row indices + the function values themselves
		row_indices.append(np.nonzero(varphi_j)[0])
		beta_image.append(np.ravel(varphi_j[row_indices[-1]]))

	## Use a CSC-sparse matrix to represent the partition of unity
	row_ind = np.hstack(row_indices)
	col_ind = np.repeat(range(J), [len(subset) for subset in row_indices])
	pou = csc_matrix((np.hstack(beta_image), (row_ind, col_ind)))
	pou = diags(1/pou.sum(axis=1).A.ravel()) @ pou
	#pou /= np.sum(pou, axis = 1)

	## The final partition of unity weights elements in the cover over B
	for j in range(J):
		j_membership = np.where(pou[:,j].todense() > 0)[0]
		subset_j = cover[cover.index_set[j]]
		ind = find_where(j_membership, subset_j)
		if (np.any(ind == None)):
			raise ValueError("The partition of unity must be supported on the closure of the cover elements.")
		
	return(pou)
	

def partition_of_unity_old(a: npt.ArrayLike, centers: npt.ArrayLike, radius: np.float64, d = dist) -> csc_matrix:
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