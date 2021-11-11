## Plan: 
## Cover -> divides data into subsets, satisfying covering property, retains underlyign neighbor structures + metric (if any)  
## Local euclidean model -> just transforms each subset from the cover ; separate from cover and PoU! 
## PoU -> Takes a cover as input, a set of weights for each point, and a function yielding a set of real values indicating 
## how "close" each point is to each cover subset; if not supplied, this is inferred from the metric, otherwise a tent function 
## or something is used. Ultimately returns an (n x J) sparse matrix where each row is normalized to sum to 1. 
import array
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix, diags
from scipy.sparse.csgraph import minimum_spanning_tree,connected_components 
from .distance import *
from .utility import find_where, cartesian_product, inverse_choose, rank_comb2, unrank_comb2
from .dimred import neighborhood_graph, neighborhood_list
from .samplers import landmarks
from .polytope import sdist_to_boundary

## Optional tools 
# from .utility import cartesian_product
# from shapely.geometry import Polygon, Point
# from scipy.optimize import golden

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
# To support constructing generic partitions of unity via bump functions, covers must also have the function:
# [cover].set_distance(X: ArrayLike, index: T) -> yields *normalized* distance of X to set T, such that 
# 	 1. d(x,T) >= 0.0  <=> (non-negative)
# 	 2. d(x,T) <= 1.0  <=> (x is contained within set T)  
# The set_distance function is only needed is the partition of unity if one is not explicitly provided.
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

## Minimally, one must implement keys(), __getitem__(), and set_distance()		 
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

def bump(similarity: float, method: Optional[str] = "triangular", **kwargs):
	''' Applies a bump function to weight a given similarity measure using a variety of bump functions '''
	assert np.all(similarity <= 1.0), "Similarity must be non-negative."
	similarity = np.maximum(similarity, 0.0)
	if method == "triangular" or method == "linear": 
		s = similarity
	elif method == "quadratic":
		s = similarity**2
	elif method == "cubic":
		s = similarity**3
	elif method == "quartic":
		s = similarity**4
	elif method == "quintic":
		s = similarity**5
	elif method == "polynomial":
		p = kwargs["p"] if kwargs.has_key("p") else 2.0
		s = similarity**p
	elif method == "gaussian":
		s = np.array([np.exp(-1.0/(d**2)) if d > 0.0 else 0.0 for d in similarity])
	elif method == "logarithmic":
		s = np.log(1+similarity)
	elif isinstance(method, Callable):
		s = method(similarity)
	else: 
		raise ValueError("Invalid bump function specified.")
	return(s)

class BallCover(Cover):
	def __init__(self, x: npt.ArrayLike, radii: Union[float, npt.ArrayLike], metric = "euclidean", space: Optional[Union[npt.ArrayLike, str]] = "union"):
		''' 
		x := either (c x d) matrix of *c* points in *d* dimensions giving the centers of balls w.r.t *metric*, or a (c x n) matrix giving the 
			   precomputed distances between the *c* balls and the *n* points in the space. See details. 
		radii := scalar, or *c*-length array giving the radii associated with every ball.
		metric := string of a common metric to use, a pairwise distance function, or the string "precomputed" if *x* is a (c x n) precomputed distance matrix.
		space := the space the balls are meant to cover. If *x* is a set of ball centers, this should be an (n x d) matrix of points to cover. 
						 If the metric is "precomputed" and *x* is a (c x n) matrix of ball distances, this need not be supplied. 
		'''

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

		# ## Detect the space, if different 
		# if space != "union" and not(space is None):
		# 	if space is str: raise NotImplementedError("Haven't implemented parsing of special spaces")
		# 	self.bounds = space

		# ## Assign attributes
		# self.centers = np.asanyarray(centers)
		# self.radii = radii 
		# self.metric = metric

		# ## Default empty cover
		# # self._neighbors = csc_matrix((0, len(self.centers)))

		# ## Detect if distances were given or points
		# if is_distance_matrix(space) or is_pairwise_distances(space):
		# 	D = space[np.triu_indices(space.shape[0], 1)] if is_distance_matrix(space) else space
		# 	n, J = inverse_choose(len(space), 2), len(L)
			
		# 	for i in range(n):
		# 		for j, index in enumerate(self.landmarks):
		# 			if i == index:
		# 				x.append(10*np.finfo(float).eps)
		# 				ri.append(i)
		# 				ci.append(j)
		# 			else:
		# 				d_ij = space[rank_comb2(i,index,n)]
		# 				if d_ij <= self.cover_radius:
		# 					x.append(d_ij)
		# 					ri.append(i)
		# 					ci.append(j)
		# 	self._neighbors = csc_matrix((x, (ri,ci)), shape=(n, J))
		# elif is_distance_matrix(space):
		# 	D = space[np.triu_indices(space.shape[0], 1)]
		# 	n, J = inverse_choose(len(D), 2), len(self.landmarks)
		# 	x, ri, ci = [], [], []
		# 	for i in range(n):
		# 		for j, index in enumerate(self.landmarks):
		# 			if i == index:
		# 				x.append(10*np.finfo(float).eps)
		# 				ri.append(i)
		# 				ci.append(j)
		# 			else:
		# 				d_ij = D[rank_comb2(i,index,n)]
		# 				if d_ij <= self.cover_radius:
		# 					x.append(d_ij)
		# 					ri.append(i)
		# 					ci.append(j)
		# 	self._neighbors = csc_matrix((x, (ri,ci)), shape=(n, J))
		# elif is_point_cloud(space):
		# 	space = np.asanyarray(space)
		# 	centers = space[np.array(self.landmarks),:]
		# 	super().construct(space) ## Postcondition: the cover must be constructed!
		# else: 
		# 	raise ValueError("Unknown input to 'space' supplied; expecting a point cloud matrix, distance matrix, or set of pairwise distances")
		
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
	
	## TODO: make efficient/demand arbitrary metrics behave in a vectorized fashion
	def set_distance(self, a: npt.ArrayLike, index: int):
		''' Returns normalized distance  such that (d(a, index) <= 1.0) -> a is within closure(cover[index]) '''
		#dx = dist(a, self.centers[[index], :], metric=self.metric)
		if is_index_like(a):
			dx = self._neighbors[a, index].A
		else:
			raise NotImplementedError("Haven't implemented querying yet")
		return(dx / self.radius(index))

## This is just a specialized ball cover w/ a fixed radius
class LandmarkCover(BallCover):
	def __init__(self, space: npt.ArrayLike, k: int, scale: float = 1.0, metric = "euclidean", **kwargs): 
		''' 
		k := the number of landmark points to use
		space := the space the balls are meant to cover. Either be an (n x d) matrix representing a point set in R^d, a (c x n) set of distances. 
		metric := string or Callable indicating the type of metric distance to use in defining the landmarks. 
		... := additional keyword arguments are passed to the landmarks(...) function. 
		'''
		assert k >= 2, "Number of landmarks must be at least 2."
		assert scale >= 1.0, "Scale must be >= 1"
		L, R  = landmarks(space, k, metric=metric, **kwargs) # here space can be dist_like or point cloud
		self.landmarks = L
		if is_dist_like(space):
			n = space.shape[0] if is_distance_matrix(space) else inverse_choose(len(space), 2)
			LD = subset_dist(space, (L, range(n)))
			super().__init__(LD, radii=np.min(R)*scale, metric="precomputed")
		else:
			assert is_point_cloud(space)
			super().__init__(x=space[L,:], radii=np.min(R)*scale, metric=metric, space=space)
		
	def __len__(self) -> int:
		return(len(self.landmarks))

	# def set_distance(self, a: npt.ArrayLike, index: int):
	# 	''' Returns normalized distance  such that (d(a, index) <= 1.0) -> a is within closure(cover[index]) '''
	# 	dx = dist(a, self.centers[[index], :], metric=self.metric)
	# 	return(dx / self.cover_radius)

class IntervalCover(Cover):
	'''
	A Cover is *CoverLike*
	Cover -> divides data into subsets, satisfying covering property, retains underlying neighbor structures + metric (if any) 
	Parameters: 
		space := the type of space to construct a cover on. Can be a point set, a set of intervals (per column) indicating the domain of the cover, 
						 or a string indicating a common configuration space. 
		n_sets := number of cover elements to use to create the cover 
		overlap := proportion of overlap between adjacent sets per dimension, as a number in [0,1). It is recommended, but not required, to keep this parameter below 0.5. 
		implicit := optional, whether to store all the subsets in a precomputed dictionary, or construct them on-demand. Defaults to the former.
	'''
	def __init__(self, a: ArrayLike, n_sets: List[int], overlap: List[float], metric: str = "euclidean", implicit: bool = False, space: Optional[Union[npt.ArrayLike, str]] = None, **kwargs):

		## Reshape a to always be a matrix of points, even if 1-dimensional
		a = np.asanyarray(a)
		if a.ndim == 1: a = a.reshape((len(a), 1))

		## If space is specified, check inputs
		if not(space is None):
			if space is str: raise NotImplementedError("Haven't implemented parsing of special spaces")
			space = np.asanyarray(space)
			if space.ndim == 1: space = np.reshape(space, (len(space), 1))
			if space.ndim != 2: raise ValueError("Invalid data shape. Must be a matrix.")
			self.dimension = space.shape[1]
			self.bbox = space
			self._data = a
			# if space.shape[0] == 2: 
			# 	self.bbox = space
			# 	self._data = a
			# else: 
			# 	self.bbox = np.vstack((np.amin(space, axis=0), np.amax(space, axis=0)))
			# 	self._data = space
		else: 
			nsets = 1 if isinstance(n_sets, int) else len(n_sets)
			nover = 1 if isinstance(overlap, float) else len(overlap)
			self.dimension = np.max([nsets, nover, a.shape[1]])
			self.bbox = np.vstack((np.amin(a, axis=0), np.amax(a, axis=0)))
			# self.bbox = np.repeat([0,1], self.dimension, axis=0).reshape((2, self.dimension))
			self._data = a

		## Set intrinsic properties of the cover
		self.metric = metric
		self.implicit = implicit
		self.n_sets = np.repeat(n_sets, self.dimension) if (isinstance(n_sets, int) or len(n_sets) == 1) else np.array(n_sets)
		self.overlap = np.repeat(overlap, self.dimension) if (isinstance(overlap, float) or len(overlap) == 1) else np.array(overlap)
		assert np.all([self.overlap[i] < 1.0 for i in range(self.dimension)]), "Proportion overlap must be < 1.0"

		## Assert these parameters match the dimension of the cover prior to assignment
		if len(self.n_sets) != self.dimension or len(self.overlap) != self.dimension:
			raise ValueError("Dimensions mismatch: supplied set or overlap arity does not match dimension of the cover.")
		self.base_width = np.diff(self.bbox, axis = 0)/self.n_sets
		self.set_width = self.base_width + (self.base_width*self.overlap)/(1.0 - self.overlap)
		
		## Choose how to handle edge orientations
		# self.gluing = np.repeat(Gluing.NONE, self.dimension) if gluing is None else gluing
		# if len(self.gluing) != self.dimension:
		# 	raise ValueError("Invalid gluing vector given. Must match the dimension of the cover.")
		
		## If implicit = False, then construct the sets and store them
		## otherwise, they must be constructed on demand via __call__
		if not(self.implicit) and not(self._data is None): 
			self.sets = self.construct(self._data)

	# def _diff_to(self, x: npt.ArrayLike, point: npt.ArrayLike):
	# 	i_dist = np.zeros(x.shape)
	# 	for d_i in range(self.dimension):
	# 		diff = abs(x[:,d_i] - point[:,d_i])
	# 		if self.gluing[d_i] == 0:
	# 			i_dist[:,d_i] = diff
	# 		elif self.gluing[d_i] == 1:
	# 			rng = abs(self.bbox[1:2,d_i] - self.bbox[0:1,d_i])
	# 			i_dist[:,d_i] = np.minimum(diff, abs(rng - diff))
	# 		elif self.gluing[d_i] == -1:
	# 			rng = abs(self.bbox[1:2,d_i] - self.bbox[0:1,d_i])
	# 			coords = self.bbox[1:2,d_i] - x[:,d_i] # Assume inside the box
	# 			diff = abs(coords - point[d_i])
	# 			i_dist[:,d_i] = np.minimum(diff, abs(rng - coords))
	# 		else:
	# 			raise ValueError("Invalid value for gluing parameter")
	# 	return(i_dist)

	
	def keys(self):
		return(np.ndindex(*self.n_sets))

	def values(self):
		if self.implicit:
			for index in self.keys():
				yield self[index]
		else: 
			for ind in self.sets.values():
				yield ind

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
		return("Interval Cover with {} sets and {} overlap over the domain {}".format(self.n_sets, self.overlap, domain_str))
	
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
	
	def set_contains(self, a: npt.ArrayLike, index: Tuple) -> ArrayLike:
		membership = np.zeros(a.shape[0], dtype="bool")
		membership[self.construct(a, index)] = True 
		return(membership)
		# eps = self.base_width/2.0
		# centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + eps
		# C = np.zeros(a.shape[0], dtype="bool")
		# for i, x in enumerate(a): 
		# 	inside = np.array([(xj >= centroid[j]-eps[j]) and (xj <= centroid[j]+eps[j]) for j, xj in enumerate(x)])
		# 	C[i] = np.all(inside)
		# return(C)

	def set_distance(self, a: npt.ArrayLike, index: Tuple) -> ArrayLike:
		assert index in list(self.keys()), "Invalid index given"
		a = np.asanyarray(a)
		if self.dimension == 1:
			eps = self.base_width/2.0
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + eps
			# diff = self._diff_to(np.asanyarray(a), centroid)
			# return(diff / eps)
			diff = np.ravel(dist(a, centroid, metric=self.metric)) / (self.set_width/2.0)
			return(diff.flatten()) ## must be flattened array
		elif self.dimension == 2: 
			eps = self.base_width/2.0
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + eps
			width = self.set_width/2
			P = centroid + cartesian_product(np.tile([-1,1], (self.dimension,1)))[[0,1,3,2],:] * width
			D = sdist_to_boundary(a, P).flatten()
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

def CircleCover(IntervalCover, Cover):
	# m_dist = lambda x,y: np.minimum(abs(x - y), (np.pi) - abs(x - y))
	def __init__(a: ArrayLike, n_sets: int, overlap: float, lb = 0.0, ub = 2*np.pi):
		assert a.ndim == 1
		self.lb = lb
		self.ub = ub 
		super().__init__(a, n_sets=n_sets, overlap=overlap)
		self.sets = self.construct(self._data)
	
	def set_distance(self, a: npt.ArrayLike, index: Tuple) -> ArrayLike:
		eps = self.base_width/2.0
		centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + eps
		diff = np.minimum(np.abs(a - centroid), self.ub-np.abs(a - centroid)) / (self.set_width/2.0)
		return(diff.flatten()) ## must be flattened array

	def construct(self, a: npt.ArrayLike, index: Optional[npt.ArrayLike] = None):
		if index is not None:
			centroid = self.bbox[0:1,:] + (np.array(index) * self.base_width) + self.base_width/2.0
			diff = np.minimum(np.abs(a - centroid), (2*np.pi)-np.abs(a - centroid))
			return(np.ravel(np.where(np.bitwise_and.reduce(diff <= self.set_width/2.0, axis = 1))))
		else:
			cover_sets = { index : self.construct(a, index) for index in self.keys() }
			return(cover_sets)

## A partition of unity can be constructed on:
## 	1. a set of balls within a metric space of arbitrary dimension, all w/ some fixed radius
##  2. a set of balls within a metric space of arbitrary dimension, each w/ a ball-specific radius
##  3. a set of convex polytopes within a metric space of arbitrary dimension
##  4. a set of polygons (possibly non-convex) /within a metric space (2D only)
##  5. a set of arbitrary geometries within a metric space (user-supplied)


def validate_cover(m, cover):
	membership = np.zeros(m, dtype=bool)
	for ind in cover.values(): 
		membership[ind] = True
	return(np.all(membership == True))


def dist_to_balls(a: ArrayLike, B: ArrayLike, R: ArrayLike, metric: str = "euclidean"):
	a, B = np.asanyarray(a), np.asanyarray(B)
	if isinstance(R, float) or len(R) == 1:
		## Case 1
		return(dist(a, B, metric=metric)/R)
	else:
		## Case 2 
		return(dist(a, B, metric=metric)/R) # broadcast?


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
## B := point cloud from some topological space
## phi := function mapping a subset of m points to (m x J) matrix 
## cover := CoverLike (supports set_contains, values)
## similarity := string or Callable 
## weights := not implemented
def partition_of_unity(B: npt.ArrayLike, cover: CoverLike, similarity: Union[str, Callable] = "triangular", weights: Optional[npt.ArrayLike] = None, check_subordinate=False) -> csc_matrix:
	if (B.ndim != 2): raise ValueError("Error: filter must be matrix.")
# assert B.shape[1] == cover.dimension, "Dimension of point set given to PoU differs from cover dimension."
	assert similarity is not None, "similarity map must be a real-valued function, or a string indicating one of the precomputed ones."
	assert isinstance(cover, CoverLike), "cover must be CoverLike"
	assert "set_distance" in dir(cover), "cover must be set_distance() function to construct a partition of unity."

	# Apply the phi map to each subset, collecting the results into lists
	J = len(cover)
	row_indices, beta_image = [], []
	for i, (index, subset) in enumerate(cover.items()): 
		## Use normalized set distance to construct partition of unity
		# dx = cover.set_distance(B[np.array(subset),:], index)
		if isinstance(cover, IntervalCover):
			dx = cover.set_distance(B[np.array(subset),:], index)
		else:
			dx = cover.set_distance(subset, index)
		sd = np.maximum(0.0, 1.0 - dx) ## todo: fix w/ bump functions

		## Record non-zero row indices + the function values themselves
		ind = np.nonzero(sd)[0]
		row_indices.append(subset[ind])
		beta_image.append(np.ravel(sd[ind]).flatten())

	## Use a CSC-sparse matrix to represent the partition of unity
	row_ind = np.hstack(row_indices)
	col_ind = np.repeat(range(J), [len(subset) for subset in row_indices])
	pou = csc_matrix((np.hstack(beta_image), (row_ind, col_ind)))
	pou = csc_matrix(pou / pou.sum(axis=1)) 

	## This checks the support(pou) \subseteq closure(cover) property
	## Only run this when not profiling
	if check_subordinate:
		for j, index in enumerate(cover.keys()):
			j_membership = np.where(pou[:,j].todense() > 0)[0]
			ind = find_where(j_membership, cover[index])
			if (np.any(ind == None)):
				raise ValueError("The partition of unity must be supported on the closure of the cover elements.")

	## Return both the partition of unity and the iota (argmax) bijection 
	return(pou)


		# if isinstance(pou, str):
		# 	## In this case, cover must have a set_distance(...) function!
		# 	## This is where the coordinates of B are needed!
		# 	self.pou = partition_of_unity(B, cover = self.cover, similarity = pou) 
		# elif issparse(pou): 
		# 	if pou.shape[1] != len(self.cover):
		# 		raise ValueError("Partition of unity must have one column per element of the cover")
		# 	for j, index in enumerate(self.cover.keys()):
		# 		pou_nonzero = np.where(pou[:,j].todense() > 0)[0]
		# 		is_invalid_pou = np.any(find_where(pou_nonzero, self.cover[index]) is None)
		# 		if (is_invalid_pou):
		# 			raise ValueError("The partition of unity must be supported on the closure of the cover elements.")
		# else: 
		# 	raise ValueError("Invalid partition of unity supplied. Must be either a string or a csc_matrix")

		# 	## In this case, cover must have a set_distance(...) function!
		# 	## This is where the coordinates of B are needed!
		# 	 = 
		# elif issparse(pou): 
		# 	if pou.shape[1] != len(self.cover):
		# 		raise ValueError("Partition of unity must have one column per element of the cover")
		# 	for j, index in enumerate(self.cover.keys()):
		# 		pou_nonzero = np.where(pou[:,j].todense() > 0)[0]
		# 		is_invalid_pou = np.any(find_where(pou_nonzero, self.cover[index]) is None)
		# 		if (is_invalid_pou):
		# 			raise ValueError("The partition of unity must be supported on the closure of the cover elements.")
		# else: 
		# 	raise ValueError("Invalid partition of unity supplied. Must be either a string or a csc_matrix")