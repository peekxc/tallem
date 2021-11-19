## cover.py
import array
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix, diags
from scipy.sparse.csgraph import minimum_spanning_tree,connected_components 
from .distance import *
from .distance import is_dist_like
from .utility import find_where, cartesian_product, inverse_choose, rank_comb2, unrank_comb2
from .dimred import neighborhood_graph, neighborhood_list
from .samplers import landmarks
from .polytope import sdist_to_boundary

## Type tools 
## NOTE: put collections first before typing!
from collections.abc import * 
from typing import *
from itertools import combinations, product

## --- Cover protocol type (its structural subtype) --- 
#
# a *Cover* is some type supporting the mixins of the Mapping[T, Union[Sequence, ArrayLike]] type
#  	- Supports mapping mixins (__getitem__, __iter__, __len__, keys, items, values)
#  	- The key type T = TypeVar('T') can be any type; i.e. any index set can be used
#  	- If value is a Sequence type => supports mixins (__getitem__, __len__, __contains__, __iter__, __reversed__, .index(), and .count()), and is sorted (precondition)!
# 	- If value is a ArrayLike type => .index() and .count() don't exist, np.searchsorted() and np.sum(...) are used instead
# A cover has the postcondition that, upon finishing its initialization from __init__(...), it must be a *valid* cover, i.e.
# if the space covers *n* points, then *validate_cover(n, cover)* must be true. 
# 
# To support constructing generic partitions of unity via bump functions, covers can optionally support bump functions
# [cover].bump(X: ArrayLike, index: T) -> yields *normalized* distance of X to set T, such that 
# 	 1. d(x,T) >= 0.0  <=> (non-negative)
# 	 2. d(x,T) <= 1.0  <=> (x is contained within set T)  
# The bump function is only needed is the partition of unity if one is not explicitly provided as a sparse matrix.
IndexArray = Union[Sequence, ArrayLike]

T = TypeVar('T')
@runtime_checkable
class CoverLike(Collection[Tuple[T, IndexArray]], Protocol): 
	def __getitem__(self, key: T) -> IndexArray: ...
	def __iter__(self): ...
	def __len__(self) -> int: ...
	def __contains__(self, key: T) -> bool: ...
	def keys(self) -> Iterator[T] : ...
	def values(self) -> Iterator[IndexArray] : ...
	def items(self) -> Iterator[Tuple[T, IndexArray]]: ...
	def set_contains(self, X: ArrayLike, index: T): ...
	# def set_distance(self, X: ArrayLike, index: T): ...

## Minimally, one must implement keys(), __getitem__()
## Must implement set_distance or set_barycenter....
class Cover(CoverLike):
	def __getitem__(self, key: T) -> IndexArray: ...
	def __iter__(self): 
		return(iter(self.keys()))
	def __len__(self) -> int: 
		return(len(list(self.keys())))
	def __contains__(self, key: T) -> bool: 
		return(list(self.keys()).contains(key))
	def keys(self) -> Iterator[T] : ...
	def values(self) -> Iterator[IndexArray] : 
		for j in self.keys(): 
			yield self[j]
	def items(self) -> Iterator[Tuple[T, IndexArray]]: 
		return(zip(self.keys(), self.values()))
	def set_distance(self, X: ArrayLike, index: T): 
		raise NotImplementedError("This cover has not defined a set distance function.")

def is_uniform(cover):
	## It happens enough that cover's are poorly constructed. This does a simple check on that. 
	open_sizes = np.array([len(subset) for subset in cover.values()])
	_p = open_sizes / np.sum(open_sizes)
	_p_entropy = -np.sum(_p*np.log(_p))
	return(_p_entropy >= 1.0 or np.all(open_sizes > 2))

def mollify(x: float, method: Optional[str] = "identity"):
	''' 
	Applies a mollifier to modify the shape of 'x' (typically the result of a bump function). 
	This method sets all negative values, if they exist, in 'x' to 0. Choices for 'method' include: 
	
	method \in ['identity', 'quadratic', 'cubic', 'quartic', 'quintic', 'gaussian', 'logarithmic']

	Alternatively, 'method' may be an arbitrary Callable. 
	
	Note that all negative entries of 'x' are always set to 0, even if a Callable is given. 
	'''
	x = np.maximum(x, 0.0) # truncate to only be positive!
	if method == "triangular" or method == "linear" or method == "identity": 
		s = x
	elif method == "quadratic":
		s = x**2
	elif method == "cubic":
		s = x**3
	elif method == "quartic":
		s = x**4
	elif method == "quintic":
		s = x**5
	elif method == "gaussian":
		s = np.array([np.exp(-1.0/(d**2)) if d > 0.0 else 0.0 for d in x])
	elif method == "logarithmic":
		s = np.log(1.0+x)
	elif isinstance(method, Callable):
		s = method(x)
	else: 
		raise ValueError("Invalid bump function specified.")
	return(s)

class BallCover(Cover):
	def __init__(self, balls: npt.ArrayLike, radii: Union[float, npt.ArrayLike], metric = "euclidean", space: Optional[Union[npt.ArrayLike, str]] = "union"):
		''' 
		balls := either (c x d) matrix of *c* points in *d* dimensions giving the centers of balls w.r.t *metric*, or a (c x n) matrix giving the 
			   precomputed distances between the *c* balls and the *n* points in the space. See details. 
		radii := scalar, or *c*-length array giving the radii associated with every ball.
		metric := string of a common metric to use, a pairwise distance function, or the string "precomputed" if *x* is a (c x n) precomputed distance matrix.
		space := the space the balls are meant to cover. If *x* is a set of ball centers, this should be an (n x d) matrix of points to cover. 
						 If the metric is "precomputed" and *x* is a (c x n) matrix of ball distances, this need not be supplied. 
		'''
		x = balls
		if metric == "precomputed":
			assert isinstance(x, np.ndarray) and x.ndim == 2
			c, n = x.shape
			if isinstance(radii, float):
				self.radii = radii
				radii = np.repeat(radii, c)
			else:
				self.radii = radii
			assert isinstance(radii, np.ndarray) and len(radii) == c
			# use array's for amortized O(1) appends + move semantics
			R, C, D = array.array('I'), array.array('I'), array.array('f') 
			for j in range(c):
				ind = np.flatnonzero(x[j,:] <= radii[j])
				R.extend(ind)
				C.extend(np.repeat(j, len(ind)))
				D.extend(x[j,ind])
			rows = np.frombuffer(R, dtype=np.int32)
			cols = np.frombuffer(C, dtype=np.int32)
			data = np.frombuffer(D, dtype=np.float32)
			self._neighbors = sp.coo_matrix((data, (rows, cols)), shape=(n, c)).tocsc()
			self.metric = "precomputed"
			# metric == "precomputed" => x is (c x n) matrix of type LD
			self._x = None 
		else: 
			assert isinstance(x, np.ndarray) and isinstance(space, np.ndarray)
			assert x.shape[1] == space.shape[1]
			n, c, d = space.shape[0], x.shape[0], x.shape[1]
			if isinstance(radii, float):
				self.radii = radii
				radii = np.repeat(radii, c)
			else:
				self.radii = radii
			assert isinstance(radii, np.ndarray) and len(radii) == c
			self._neighbors = neighborhood_list(centers=x, a=space, radius=radii, metric=metric).tocsc()
			self.metric = metric
			# space of type P => x is (c x d) matrix of type P representing ball centers
			self._x = x 	
			
	def radius(self, index: int):
		assert index >= 0 and index < len(self), "Invalid index given."
		return(self.radii if isinstance(self.radii, float) else self.radii[index])

	def __getitem__(self, index: int):
		assert index < len(self), "Invalid index supplied. Must be integer less than the number of cover subsets."
		# return(self._neighbors.indices[self._neighbors.indptr[index]:self._neighbors.indptr[index]])
		return(self._neighbors.getcol(index).indices)

	def keys(self):
		return(range(len(self)))

	def __len__(self) -> int:
		return(self._neighbors.shape[1])
	
	def bump(self, a: npt.ArrayLike, index: int, normalize: bool = False):
		''' Returns normalized distance  such that (d(a, index) <= 1.0) -> a is within closure(cover[index]) '''
		if is_index_like(a):
			dx = self._neighbors[a, index].A
			return(self.radius(index) - dx)
		elif is_point_cloud(a) and is_point_cloud(self._x):
			dx = dist(self._x[index,:], a, metric=self.metric)
			return(self.radius(index) - dx)
		else:
			raise NotImplementedError("Unknown distance/point combination passed.")
		return(dx / self.radius(index) if normalize else dx)

## This is just a specialized ball cover w/ a fixed radius
class LandmarkCover(BallCover):
	def __init__(self, space: npt.ArrayLike, n_sets: int, scale: float = 1.0, metric = "euclidean", **kwargs): 
		''' 
		space := the space the balls are meant to cover. Either be an (n x d) matrix representing a point set in R^d or a distance-like object. 
		n_sets := the number of sets in the cover. Equivalently, the number of landmark points to use.
		scale := relative scaling of cover sets. Must be >= 1.0.
		metric := string or Callable indicating the type of metric distance to use in defining the landmarks. 
		... := additional keyword arguments are passed to the landmarks(...) function. 
		'''
		assert n_sets >= 2, "Number of landmarks must be at least 2."
		assert scale >= 1.0, "Scale must be >= 1"
		L, R  = landmarks(space, n_sets, metric=metric, **kwargs) # here space can be dist_like or point cloud
		self.landmarks = L
		if is_dist_like(space):
			n = space.shape[0] if is_distance_matrix(space) else inverse_choose(len(space), 2)
			LD = subset_dist(space, (L, range(n)))
			super().__init__(LD, radii=np.min(R)*scale, metric="precomputed")
		else:
			assert is_point_cloud(space)
			super().__init__(space[L,:], radii=np.min(R)*scale, metric=metric, space=space)
		
	def __len__(self) -> int:
		return(len(self.landmarks))

class IntervalCover(Cover):
	'''
	A Cover is *CoverLike*
	Cover -> divides data into subsets, satisfying covering property, retains underlying neighbor structures + metric (if any) 
	Parameters: 
		a := 
		n_sets := number of cover elements to use to create the cover 
		scale := 
		bounds := (optional) bounds indicating the domain of the cover. Can be given as a set of intervals (per column). Otherwise the bounding box is inferred from 'a'.  
		implicit := optional, whether to store all the subsets in a precomputed dictionary, or construct them on-demand. Defaults to the former.
	'''
	def __init__(self, a: ArrayLike, n_sets: List[int], scale: List[float], implicit: bool = False, bounds: Optional[Union[npt.ArrayLike, str]] = None, **kwargs):

		## Reshape a to always be a matrix of points, even if 1-dimensional
		a = np.asanyarray(a)
		if a.ndim == 1: a = a.reshape((len(a), 1))

		## If bounds is specified, check inputs
		if not(bounds is None):
			if bounds is str: raise NotImplementedError("Haven't implemented parsing of special spaces")
			bounds = np.asanyarray(bounds)
			if bounds.ndim == 1: bounds = np.reshape(bounds, (len(bounds), 1))
			if bounds.ndim != 2: raise ValueError("Invalid data shape. Must be a matrix.")
			self.dimension = bounds.shape[1]
			self.bbox = bounds
			self._data = a
		else: 
			nsets = 1 if isinstance(n_sets, int) else len(n_sets)
			nover = 1 if isinstance(scale, float) else len(scale)
			self.dimension = np.max([nsets, nover, a.shape[1]])
			self.bbox = np.vstack((np.amin(a, axis=0), np.amax(a, axis=0)))
			self._data = a

		## Set intrinsic properties of the cover
		self.implicit = implicit
		self.n_sets = np.repeat(n_sets, self.dimension) if (isinstance(n_sets, int) or len(n_sets) == 1) else np.array(n_sets)
		self.scale = np.repeat(scale, self.dimension) if (isinstance(scale, float) or len(scale) == 1) else np.array(scale)
		# self.overlap = np.repeat(overlap, self.dimension) if (isinstance(overlap, float) or len(overlap) == 1) else np.array(overlap)
		# assert np.all([self.overlap[i] < 1.0 for i in range(self.dimension)]), "Proportion overlap must be < 1.0"

		## Assert these parameters match the dimension of the cover prior to assignment
		if len(self.n_sets) != self.dimension or len(self.scale) != self.dimension:
			raise ValueError("Dimensions mismatch: supplied set or scale arity does not match dimension of the cover.")
		self.base_width = np.diff(self.bbox, axis = 0)/self.n_sets
		self.set_width = self.base_width*self.scale
		# self.set_width = self.base_width + (self.base_width*self.overlap)/(1.0 - self.overlap)
		
		## If implicit = False, then construct the sets and store them
		## otherwise, they must be constructed on demand via __call__
		if not(self.implicit) and not(self._data is None): 
			self.sets = self.construct(self._data)
	
	def keys(self):
		return(np.ndindex(*self.n_sets))

	# This is custom to allow the representation to be implicit
	def values(self):
		if self.implicit:
			for index in self.keys():
				yield self[index]
		else: 
			for ind in self.sets.values():
				yield ind

	# This is custom to allow the representation to be implicit
	def __getitem__(self, index):
		if isinstance(index, tuple):
			return(self.construct(self._data, index) if self.implicit else self.sets[index])
		elif int(index) == index:
			index = list(self.sets.keys())[index]
			return(index, self[index])
		else: 
			raise ValueError("Cover must be indexed by either a tuple or an integer index")
	
	def __len__(self):
		return(np.prod(self.n_sets))

	def __repr__(self) -> str:
		domain_str = " x ".join(np.apply_along_axis("{}".format, axis=0, arr=self.bbox))
		return("Interval Cover with {} sets and {} scale over the domain {}".format(self.n_sets, self.scale, domain_str))
	
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
	
	def set_contains(self, a: npt.ArrayLike, index: Tuple) -> ArrayLike:
		membership = np.zeros(a.shape[0], dtype="bool")
		membership[self.construct(a, index)] = True 
		return(membership)

	def bump(self, a: npt.ArrayLike, index: Tuple, normalize=False) -> ArrayLike:
		assert index in list(self.keys()), "Invalid index given"
		a = np.asanyarray(a)
		if self.dimension == 1:
			eps = self.base_width/2.0
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + eps
			lb = centroid - self.set_width/2.0
			ub = centroid + self.set_width/2.0
			diff = np.minimum(abs(a-lb), abs(a-ub)).flatten()
			diff = diff * np.array([1.0 if x else -1.0 for x in np.bitwise_and(a >= lb, a <= ub)])
			return(diff/(abs(self.set_width/2.0)) if normalize else diff)
		elif self.dimension == 2:
			eps = self.base_width/2.0
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + eps
			width = self.set_width/2
			P = centroid + cartesian_product(np.tile([-1,1], (self.dimension,1)))[[0,1,3,2],:] * width
			D = sdist_to_boundary(a, P).flatten()
			if not(normalize):
				return(D.flatten())
			else: 
				max_eps = sdist_to_boundary(centroid, P).item()
				return(D.flatten()/max_eps)
		else: 
			raise NotImplementedError("Interval cover only supports dimensions up to 2")

	def construct(self, a: npt.ArrayLike, index: Optional[npt.ArrayLike] = None):
		if index is not None:
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + self.base_width/2.0
			diff = np.abs(a - centroid)
			return(np.ravel(np.where(np.bitwise_and.reduce(diff <= self.set_width/2.0, axis = 1))))
			# # diff = self.set_distance(a, centroid)
			# diff = dist(centroid, a, metric=self.metric).flatten()

			# # return(np.nonzero(self.set_distance(a, index) <= 1.0)[0])
			# return(np.flatnonzero(diff <= self.set_width/2.0))
		else:
			cover_sets = { index : self.construct(a, index) for index in self.keys() }
			return(cover_sets)

# class AutoCover:
# 	def __init__(self):
# 		self.x = "x"


## Minimally, one must implement keys(), __getitem__(), and set_distance()	
class CircleCover(Cover):
	''' 
	1-dimensional circular cover over the interval [lb, ub] (defaults to [0, 2*pi]) 
	This cover computes distances modulo the givens bounds  
	'''
	def __init__(self, a: ArrayLike, n_sets: int, scale: float, lb = 0.0, ub = 2*np.pi):
		if not(isinstance(a, np.ndarray)):
			a = np.asanyarray(a)
		assert a.ndim == 1 or (len(a) == np.prod(a.shape))
		assert lb < ub
		assert scale >= 1.0
		self.lb = lb
		self.ub = ub 
		self.n_sets = n_sets
		self.scale = scale
		self.base_width = abs(ub-lb)/n_sets
		self.set_width = self.base_width*scale
		self.sets = self.construct(a)
		self._x = a # save points
	
	def keys(self) -> Iterable: 
		return(range(self.n_sets))
	
	def __len__(self) -> int: 
		return(self.n_sets)

	def __getitem__(self, index: int): 
		return(self.sets[index])

	def bump(self, a: npt.ArrayLike, index: int, normalize: bool = False) -> ArrayLike:
		assert index in self.sets.keys()
		if is_index_like(a):
			return(self.bump(self._x[a], index, normalize))
		elif is_point_cloud(a) or (isinstance(a, np.ndarray) and a.ndim == 1):
			a = np.reshape(a, (len(a), 1))
			eps = self.base_width/2.0
			centroid = self.lb + (index * self.base_width) + self.base_width/2.0
			diff = np.minimum(np.abs(a - centroid), self.ub-np.abs(a - centroid)).flatten()
			sgn = np.array([1.0 if x else -1.0 for x in (diff <= self.set_width/2.0)])
			diff = diff * sgn
			return(diff/(self.set_width/2.0) if normalize else diff)
		else: 
			raise NotImplementedError("Unsupported input type for bump function")

	def construct(self, a: npt.ArrayLike, index: int = None):
		if index is not None:
			centroid = self.lb + (index * self.base_width) + self.base_width/2.0
			diff = np.minimum(np.abs(a - centroid), self.ub - np.abs(a - centroid))
			return(np.flatnonzero(diff <= self.set_width/2.0))
		else:
			cover_sets = { index : self.construct(a, index) for index in self.keys() }
			return(cover_sets)

def validate_cover(m, cover):
	membership = np.zeros(m, dtype=bool)
	for ind in cover.values(): 
		membership[ind] = True
	return(np.all(membership == True))

def dist_to_boundary(P: npt.ArrayLike, x: npt.ArrayLike):
	''' Given ordered vertices constituting polygon boundary and a point 'x', determines distance from 'x' to polygon boundary on ray emenating from centroid '''
	
	return(sdist_to_boundary(x, P))
	# B = Polygon(P)
	# c = B.centroid
	
	# ## direction away from centroid 
	# v = np.array(x) - np.array(c.coords) 
	# v = v / np.linalg.norm(v)
	
	# ## max possible distance away 
	# xx, yy = B.minimum_rotated_rectangle.exterior.coords.xy
	# max_diam = np.max(np.abs(np.array(xx) - np.array(yy)))

	# ## minimize convex function w/ golden section search 
	# dB = lambda y: B.boundary.distance(Point(np.ravel(c + y*v)))
	# y_opt = golden(dB, brack=(0, max_diam))

	# ## return final distance
	# eps = np.linalg.norm(np.array(c.coords) - np.ravel(c + y_opt*v))
	# dc = np.linalg.norm(np.array(x) - np.array(c.coords))
	# return(eps, dc)




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



## A Partition of unity is 
## phi := function mapping a subset of m points to (m x J) matrix 
## cover := CoverLike that has a .bump() function
## mollifer := string or Callable indicating the choice of mollifier
## weights := not implemented
## normalize := whether to request the bump functions normalize their output. Defaults to true. 
def partition_of_unity(cover: CoverLike, mollifier: Union[str, Callable] = "identity", weights: Optional[npt.ArrayLike] = None, check_subordinate=False, normalize=True) -> csc_matrix:
	assert mollifier is not None, "mollifier must be a real-valued function, or a string indicating one of the precomputed ones."
	assert isinstance(cover, CoverLike), "cover must be CoverLike"
	assert "bump" in dir(cover), "cover must be bump() function to construct a partition of unity with a mollifier."

	# Apply the phi map to each subset, collecting the results into lists
	J = len(cover)
	row_ind, col_ind, phi_image = array.array('I'), array.array('I'), array.array('f') 
	for i, (index, subset) in enumerate(cover.items()): 
		## Use mollified-bump function to construct "default" partition of unity
		dx = cover.bump(subset, index, normalize) # every cover should support index-like bump functionality
		dx = mollify(dx, mollifier)

		## Record non-zero row indices + the function values themselves
		ind = np.flatnonzero(dx >= 0.0)
		row_ind.extend(subset[ind])
		col_ind.extend(np.repeat(i, len(subset)))
		phi_image.extend(np.ravel(dx[ind]).flatten())

	## Use a CSC-sparse matrix to represent the partition of unity
	R, C = np.frombuffer(row_ind, dtype=np.int32), np.frombuffer(col_ind, dtype=np.int32)
	P = np.frombuffer(phi_image, dtype=np.float32)
	pou = csc_matrix((P, (R, C)))
	pou = csc_matrix(pou / pou.sum(axis=1)) 

	## This checks the support(pou) \subseteq closure(cover) property
	## Only run this when not profiling
	# if check_subordinate:
	# 	for j, index in enumerate(cover.keys()):
	# 		pou.getcol(j) == 
	# 		j_membership = np.flatnonzero(pou[:,j] > 0)
	# 		ind = cover[index])
	# 		if (np.any(ind == None)):
	# 			raise ValueError("The partition of unity must be supported on the closure of the cover elements.")

	## Return both the partition of unity and the iota (argmax) bijection 
	return(pou)
