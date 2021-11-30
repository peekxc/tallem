# %% Sampler imports
import numpy as np
import itertools as it
from numpy.typing import ArrayLike
from typing import *
from .distance import *
from .utility import find_points

## Import the specific extensions module we need
from tallem.extensions import landmark

# %% Definitions
def landmarks(a: ArrayLike, k: Optional[int] = 15, eps: Optional[float] = -1.0, seed: int = 0, diameter: bool = False, metric = "euclidean"):
	'''
	Computes landmarks points for a point set or set of distance 'a' using the 'maxmin' method. 

	Parameters:
		a := (n x d) matrix of *n* points in *d* dimensions, a distance matrix, or a set of pairwise distances
		k := (optional) number of landmarks requested. Defaults to 15. 
		eps := (optional) covering radius to stop finding landmarks at. Defaults to -1.0, using *k* instead.
		seed := index of the initial point to be the first landmark. Defaults to 0.
		diameter := whether to include the diameter as the first insertion radius (see details). Defaults to False. 
		metric := metric distance to use. Ignored if *a* is a set of distances. See details. 

	Details: 
		- The first radius is always the diameter of the point set, which can be expensive to compute for high dimensions, so by default "inf" is used as the first covering radius 
		- If the 'metric' parameter is "euclidean", the point set is used directly, otherwise the pairwise distances are computed first via 'dist'
		- If 'k' is specified an 'eps' is not (or it's -1.0), then the procedure stops when 'k' landmarks are found. The converse is true if k = 0 and eps > 0. 
			If both are specified, the both are used as stopping criteria for the procedure (whichever becomes true first).
		- Given a fixed seed, this procedure is deterministic. 

	Returns a pair (indices, radii) where:
		indices := the indices of the points defining the landmarks; the prefix of the *greedy permutation* (up to 'k' or 'eps')
		radii := the insertion radii whose values 'r' yield a cover of 'a' when balls of radius 'r' are places at the landmark points.

	The maxmin method yields a logarithmic approximation to the geometric set cover problem.

	'''
	a = np.asanyarray(a)
	k = 0 if k is None else int(k)
	eps = -1.0 if eps is None else float(eps)
	seed = int(seed)
	if is_dist_like(a):
		if is_distance_matrix(a):
			a = a[np.triu_indices(a.shape[0], k=1)]
		if a.dtype != np.float64:
			a = a.astype(np.float64)
		indices, radii = landmark.maxmin(a, eps, k, True, seed)
	elif metric == "euclidean" and is_point_cloud(a):
		indices, radii = landmark.maxmin(a.T, eps, k, False, seed)
		radii = np.sqrt(radii)
	else:
		raise ValueError("Unknown input type detected. Must be a matrix of points, a distance matrix, or a set of pairwise distances.")
	
	## Check is a valid cover 
	is_monotone = np.all(np.diff(-np.array(radii)) >= 0.0)
	assert is_monotone, "Invalid metric: non-monotonically decreasing radii found."

	return(indices, radii)

	# a = np.array(a)
	# n = a.shape[0]
	# if k is None: k = n

	# ## Change metric if need be
	# pc = [Point(a[i,:]) for i in range(n)]
	# if metric != "euclidean":
	# 	# def metric_dist(self, other): 
	# 	# 	p1 = np.fromiter(self, dtype=np.float32)
	# 	# 	p2 = np.fromiter(other, dtype=np.float32) 
	# 	# 	return(dist(p1,p2,metric=metric).item())
	# 	# for p in pc: 
	# 	# 	p.dist = types.MethodType(metric_dist, p)
	# 	return(landmarks_dist(a, k, method, seed, diameter, metric))

	# ## Find the landmarks using clarksons greedy algorithm
	# landmark_iter = clarksongreedy._greedy(pc, seed=pc[seed], alpha=alpha) 
	# greedy_lm = [{ "point": p[0], "predecessor": p[1] } for p in it.islice(landmark_iter, k) ]
	# landmarks = np.array([x["point"] for x in greedy_lm])
	
	# ## Recover point row indices using an O(log(n)) search per point
	# landmark_idx = find_points(landmarks, np.array(pc))
	
	# ## Compute the cover/insertion radii
	# predecessors = np.array([landmark_idx[x["predecessor"]] for x in greedy_lm[1:]], dtype=np.int32)
	# cover_radii = dist(a[landmark_idx[1:],:], a[predecessors,:], pairwise = True, metric=metric)
	# if diameter:
	# 	from scipy.spatial import ConvexHull
	# 	hull = ConvexHull(a, qhull_options="QJ")
	# 	max_radius = np.max(dist(a[hull.vertices,:]))
	# else: 
	# 	max_radius = float('inf')
	# cover_radii = np.insert(cover_radii, 0, max_radius, axis = 0)
	# return({ "indices" : landmark_idx, "radii" : cover_radii })

# def landmarks_dist(a: ArrayLike, k: Optional[int], method: str = "maxmin", seed: int = 0, diameter: bool = False, metric = "euclidean"):
# 	L, R = np.array(seed, dtype=int), np.array([float('inf')], dtype=float)
# 	# D = dist(a, as_matrix=True, metric=metric)
# 	for ki in range(1, k):
# 		d = dist(a, a[np.array(L),:], metric=metric)
# 		d[d == 0] = float('inf')
# 		d = np.apply_along_axis(np.min, 1, d)
# 		d[d == float('inf')] = 0.0
# 		d[L] = 0.0
# 		L, R = np.append(L, np.argmax(d)), np.append(R, d[np.argmax(d)])
# 	return({ "indices" : L, "radii" : R })




def landmark_sampler(a: ArrayLike, m, k, method: str = "precomputed"):
	''' 
	Sampler which uniformly samples from k precomputed landmark permutations 
	each of size k. 
	'''
	seed_idx = np.random.randint(0, a.shape[0], size=m)
	L = [landmarks(X, k, seed = si)['indices'] for si in seed_idx]
	g = np.random.default_rng()
	def sampler(n: int):
		nonlocal k, g, m, L
		num_rotations = int(n/k)
		num_extra = n % k 
		if num_rotations == 0:
			lm_idx = g.integers(0, m, size=1)[0] 
			for idx in L[lm_idx][:num_extra]:
				yield idx
		else: 
			full_lm_idx = g.integers(0, m, size=num_rotations) 
			extra_lm_idx = g.integers(0, m, size=1)[0] 
			for lm_idx in full_lm_idx:
				for idx in L[lm_idx]:
					yield idx
			for idx in L[extra_lm_idx][:num_extra]:
				yield idx
	return(sampler)

def uniform_sampler(n: int):
	g = np.random.default_rng()
	def sampler(k: int):
		nonlocal n, g
		for s in g.integers(0, n, size=k):
			yield s
	return(sampler)

def cyclic_sampler(n: int):
	current_idx = 0
	def sampler(num_samples: int):
		nonlocal current_idx
		for _ in range(num_samples):
			yield current_idx
			current_idx = current_idx+1 if (current_idx+1) < n else 0
	return(sampler)
