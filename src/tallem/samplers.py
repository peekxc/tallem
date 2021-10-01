# %% Sampler imports
import types
import numpy as np
import itertools as it
from numpy.typing import ArrayLike
from typing import Optional
from greedypermutation import clarksongreedy
from greedypermutation.point import Point
from .distance import dist
from .utility import find_points

# %% Definitions
def landmarks(a: ArrayLike, k: Optional[int], method: str = "maxmin", seed: int = 0, alpha: np.float32 = 1.0, diameter: bool = False, metric = "euclidean"):
	'''
	Computes landmarks points for a point set 'a' 

	Note the the first radius is always the diameter of the point set, but this can be 
	expensive to compute for high dimensions, so by default "inf" is used as the first covering radius 
	'''
	a = np.array(a)
	n = a.shape[0]
	if k is None: k = n

	## Change metric if need be
	pc = [Point(a[i,:]) for i in range(n)]
	if metric != "euclidean":
		# def metric_dist(self, other): 
		# 	p1 = np.fromiter(self, dtype=np.float32)
		# 	p2 = np.fromiter(other, dtype=np.float32) 
		# 	return(dist(p1,p2,metric=metric).item())
		# for p in pc: 
		# 	p.dist = types.MethodType(metric_dist, p)
		return(landmarks_dist(a, k, method, seed, diameter, metric))

	## Find the landmarks using clarksons greedy algorithm
	landmark_iter = clarksongreedy._greedy(pc, seed=pc[seed], alpha=alpha) 
	greedy_lm = [{ "point": p[0], "predecessor": p[1] } for p in it.islice(landmark_iter, k) ]
	landmarks = np.array([x["point"] for x in greedy_lm])
	
	## Recover point row indices using an O(log(n)) search per point
	landmark_idx = find_points(landmarks, np.array(pc))
	
	## Compute the cover/insertion radii
	predecessors = np.array([landmark_idx[x["predecessor"]] for x in greedy_lm[1:]], dtype=np.int32)
	cover_radii = dist(a[landmark_idx[1:],:], a[predecessors,:], pairwise = True, metric=metric)
	if diameter:
		from scipy.spatial import ConvexHull
		hull = ConvexHull(a, qhull_options="QJ")
		max_radius = np.max(dist(a[hull.vertices,:]))
	else: 
		max_radius = float('inf')
	cover_radii = np.insert(cover_radii, 0, max_radius, axis = 0)
	return({ "indices" : landmark_idx, "radii" : cover_radii })

def landmarks_dist(a: ArrayLike, k: Optional[int], method: str = "maxmin", seed: int = 0, diameter: bool = False, metric = "euclidean"):
	L, R = np.array(seed, dtype=int), np.array([float('inf')], dtype=float)
	# D = dist(a, as_matrix=True, metric=metric)
	for ki in range(1, k):
		d = dist(a, a[np.array(L),:], metric=metric)
		d[d == 0] = float('inf')
		d = np.apply_along_axis(np.min, 1, d)
		d[d == float('inf')] = 0.0
		d[L] = 0.0
		L, R = np.append(L, np.argmax(d)), np.append(R, d[np.argmax(d)])
	return({ "indices" : L, "radii" : R })


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


def landmark_module():
	from . import landmark
	return(landmark)