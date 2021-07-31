## TALLEM __init__.py file 
## IMports all the main utilities of TALLEM and provides a wrapper for the entire function

import numpy as np
import numpy.typing as npt
from typing import Callable, Iterable, List, Set, Dict, Optional, Tuple, Any, Union, Sequence
from itertools import combinations
from scipy.sparse import csc_matrix

# from . import sc
# from . import distance 
# from . import procrustes 
# from . import mds
# from . import samplers 
# from . import stiefel 
# from . import cover
# from . import utility
# from . import assembly

# __all__ = [
# 	'function_1',
# 	'function_2'
# ]
# TODO: define __all__
from .sc import delta0D
from .distance import dist
from .procrustes import opa, align_models, global_translations
from .mds import classical_MDS
from .procrustes import opa
from .samplers import uniform_sampler
from .stiefel import frame_reduction
from .cover import IntervalCover, partition_of_unity
from .utility import find_where
from .assembly import assemble_frames


import autograd.scipy.linalg as auto_scipy 
import autograd.numpy as auto_np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent


# %% debug 
# PoU = partition_of_unity(B, cover = cover, beta = "triangular")

	## Phi map: supplying i returns the phi map for the largest 
	## value in the partition of unity; supplying j as well returns 
	## the specific phi map for the jth cover element (possibly 0)
	# def phi(i, j = None):
	# 	J = P.shape[1]
	# 	k = np.argmax(P[i,:]) if j is None else j
	# 	out = np.zeros((d*J, d))
	# 	def weighted_omega(j):
	# 		nonlocal i, k
	# 		w = np.sqrt(P[i,j])
	# 		pair_exists = np.array([pair in Omega_map.keys() for pair in [(k,j), (j,k)]])
	# 		if w == 0.0 or not(pair_exists.any()):
	# 			return(w*np.eye(d))
	# 		return(w*Omega_map[(k,j)]['rotation'] if pair_exists[0] else w*Omega_map[(j,k)]['rotation'].T)
	# 	return(np.vstack([weighted_omega(j) for j in range(J)]))

	
## Rotation, scaling, translation, and distance information for each intersecting cover subset

# from collections.abc import Iterable

class TALLEM():
	'''
	TALLEM: Topological Assembly of Locally Euclidean Models

	Parameters: 
		X := an (n x p) numpy array representing *n* points in *p* space.
		B := an (n x q) numpy array representing the image of f : X -> B, where f is a map that cpatures the topology and non-linearity of X. 
		cover := Iterable of length J that covers some topological space B, where B is the image of some map f : X -> B
		local_map := a callable mapping (m x p) subsets of X to some (m x d) space, where d < p, which approximately preserves the metric on X. 
		pou := partition of unity, either one of ['triangular', 'quadratic'], or an (n x J) ArrayLike object whose rows
					 yield weights indicating the strength of membership of that point with each set in the cover.
		
	'''
	
	def __init__(self, cover: Iterable, local_map: Callable[npt.ArrayLike, npt.ArrayLike], n_components: int = 2, pou: Union[str, csc_matrix] = "triangular"):
		if not(isinstance(cover, Iterable)): raise ValueError("cover argument must be an Iterable.")		
		
		## Store callable as part of the cover's initialization
		self.D = n_components ## Target dimension of the embedding 
		self.local_map = local_map
		
		## Validate cover 
		## TODO: use generic to generalize this to a type-erased generic! 
		if not(isinstance(cover, Iterable)):
			raise ValueError("Cover must be Iterable!")
		self.cover = cover 

		## Validate the partition of unity respects the closure condition relative to the cover 
		if isinstance(pou, csc_matrix):
			if pou.shape[1] != len(self.cover):
				raise ValueError("Partition of unity must have one column per element of the cover")
		self.pou = pou

	
	def fit(self, X: npt.ArrayLike, B: npt.ArrayLike, pou: Optional[Union[str, csc_matrix]] = None):
		X, B = np.array(X, copy=False), np.array(B, copy=False)
		if X.shape[0] != B.shape[0]: raise ValueError("X and B must have the same number of rows.")
		if X.ndim == 1: X = X.reshape((len(X), 1))
		if B.ndim == 1: B = B.reshape((len(B), 1))
		
		## Map the local euclidean models
		## Note: the extra array constructor ensures singleton subsets are reported as matrices
		self.models = { index : self.local_map(X[np.array(subset),:]) for index, subset in self.cover }

		## Construct a partition of unity
		J = len(self.cover)
		if not(pou is None): self.pou = pou
		if isinstance(self.pou, str):
			self.pou = partition_of_unity(B, cover = self.cover, beta = self.pou)
		elif isinstance(pou, csc_matrix): 
			if pou.shape[1] != len(self.cover):
				raise ValueError("Partition of unity must have one column per element of the cover")
			for i in range(J):
				pou_nonzero = np.where(pou[:,i].todense() > 0)[0]
				cover_subset = self.cover[self.cover.index_set[i]]
				is_invalid_pou = np.any(find_where(pou_nonzero, cover_subset) is None)
				if (is_invalid_pou):
					raise ValueError("The partition of unity must be supported on the closure of the cover elements.")
		else: 
			raise ValueError("Invalid partition of unity supplied. Must be either a string or a csc_matrix")

		## Align the local reference frames using Procrustes
		self.alignments = align_models(self.cover, self.models)
		
		## Get global translation vectors using cocyle condition 	
		translations = global_translations(self.cover, self.alignments)

		## Solve the Stiefel manifold optimization for the projection matrix 
		self.A0, self.A, self._stf = frame_reduction(self.alignments, self.pou, self.D, fast_gradient=False, optimize = False)

		## Assemble the frames!
		self.embedding_ = assemble_frames(self._stf, self.A, self.cover, self.pou, self.models, translations)

		return(self)

	def fit_transform(self, X: npt.ArrayLike, B: npt.ArrayLike, **fit_params) -> npt.ArrayLike:
		self.fit(X, B, **fit_params)
		return(self.embedding_)

	def __repr__(self) -> str:
		return("TALLEM instance")


## TALLEM dimennsionality reduction algorithm -- Full wrapper
## TODO: Need to supply cover options
def tallem_transform(a: npt.ArrayLike, f: npt.ArrayLike, d: int = 2, D: int = 3, J: int = 10):
	X = np.array(a)
	n = X.shape[0]

	## Form basic partition of unity
	poles = np.linspace(0, 2*np.pi, J)
	eps = 0.75*np.min(np.diff(poles))
	f_mat = np.reshape(f, (len(f), 1))
	p_mat = np.reshape(poles, (len(poles), 1))
	def angular_dist(a,b):
		ddist = dist(a,b)
		odist = np.abs(ddist - 2*np.pi)
		return(np.minimum(ddist, odist))
	P = partition_of_unity_old(f_mat, p_mat, eps, d=angular_dist)
	# cover = IntervalCover(f, n_sets = 10, overlap = 0.40, gluing=[1])
	# PoU = partition_of_unity(f, cover, beta="triangular")
	# P = PoU.todense()

	## Identify the cover set for each point
	point_cover_map = [np.ravel(P[i,:].nonzero()[0])for i in range(n)]

	## Aggregate points into cover->point map
	nc = P.shape[1] # number of cover sets
	cover_point_map = { i : np.ravel(P[:,i].nonzero()[0]) for i in range(nc) }

	## Use MDS coordinates to create the local euclidean models
	Fj = { i : [] for i in range(nc) }
	for i in range(nc):
		dx = dist(X[cover_point_map[i],:], as_matrix=True) 
		Fj[i] = classical_MDS(dx, k=d)

	## Extract intersection maps 
	def intersection_map(pair):
		i, j = pair
		ij_ind = np.intersect1d(cover_point_map[i], cover_point_map[j])
		if len(ij_ind) > 0:
			i_idx = np.ravel([np.where(np.array(cover_point_map[i]) == element) for element in ij_ind])
			j_idx = np.ravel([np.where(np.array(cover_point_map[j]) == element) for element in ij_ind])
			return({ "indices" : ij_ind, "model1" : Fj[i][i_idx,:], "model2": Fj[j][j_idx,:] })
		else: 
			v = np.empty((0,), np.float64)
			return({ "indices": np.asarray(v, dtype=np.int32), "model1": v, "model2": v })

	## Solve Procrustes problem for each non-empty intersection	
	Omega_map = {}
	edges = []
	for (i,j) in combinations(range(nc), 2):
		F_models = intersection_map((i,j))
		if len(F_models["indices"]) > 1:
			Omega_map[(i,j)] = old_procrustes(F_models["model1"], F_models["model2"], transform=False)

	## Phi map: supplying i returns the phi map for the largest 
	## value in the partition of unity; supplying j as well returns 
	## the specific phi map for the jth cover element (possibly 0)
	def phi(i, j = None):
		J = P.shape[1]
		k = np.argmax(P[i,:]) if j is None else j
		out = np.zeros((d*J, d))
		def weighted_omega(j):
			nonlocal i, k
			w = np.sqrt(P[i,j])
			pair_exists = np.array([pair in Omega_map.keys() for pair in [(k,j), (j,k)]])
			if w == 0.0 or not(pair_exists.any()):
				return(w*np.eye(d))
			return(w*Omega_map[(k,j)]['rotation'] if pair_exists[0] else w*Omega_map[(j,k)]['rotation'].T)
		return(np.vstack([weighted_omega(j) for j in range(J)]))

	## --- Optimization to find the best A matrix --- 
	def initial_frame(D, phi, n):
		Fb = np.vstack([phi(j).T for j in range(n)])
		Phi_N = Fb.T @ Fb # (dJ x dJ)
		Eval, Evec = np.linalg.eigh(Phi_N)
		return(Evec[:,np.argsort(-Eval)[:D]])

	# Need Stiefel(n=d*J,p=D) as Stiefel(n,p) := space of (n x p) orthonormal matrices
	manifold = Stiefel(d*nc, D)

	## Huber loss function to optimize
	def huber_loss(subsetter: Callable[[], Iterable[int]], epsilon: auto_np.float32 = 1.0, update_eps = lambda eps: eps):
		def cost_function(A):
			nonlocal epsilon 
			nuclear_norm = 0.0
			for j in subsetter():
				M = auto_np.array(A.T @ phi(j), dtype='float')
				svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False)
				nuclear_norm += auto_np.sum([t if t >= epsilon else (t**2)/epsilon for t in auto_np.abs(svals)])
			epsilon = update_eps(epsilon)
			return(-nuclear_norm)
		return(cost_function)

	sampler = uniform_sampler(n)
	problem = Problem(manifold=manifold, cost=huber_loss(lambda: sampler(n), 0.30))
	solver = SteepestDescent()
	A0 = initial_frame(D, phi, X.shape[0])
	Xopt = solver.solve(problem=problem, x=A0)
	
	## TODO: remove
	Xopt = A0

	## Get optimal translation vectors 
	S = (np.fromiter(range(nc), dtype=int), list(Omega_map.keys()))
	Omega = { k: v['rotation'] for k,v in Omega_map.items() }
	Trans = { k: v['translation'] for k,v in Omega_map.items() }

	## Evaluates pseudo-inverse on the coboundary matrix    
	deltaX = np.linalg.pinv(delta0D(S, Omega))
	shiftE = np.zeros(d*len(S[1]))
	for (index,(i1,i2)) in enumerate(S[1]):
			shiftE[index*d:(index+1)*d] = Trans[(i1,i2)]
	shiftV = np.matmul(deltaX,shiftE)

	## Offsets contain the translation vectors, keyed by edge indices in the nerve complex 
	offsets = {}
	for index in range(nc):
		offsets[index] = shiftV[index*d:(index+1)*d]

	## Construct the global assembly function 
	A_opt = Xopt
	A_cov = A_opt @ A_opt.T
	assembly = np.zeros((n, D), dtype=np.float64)
	for i in range(n):
		idx = np.ravel(P[i,:].nonzero())
		coords = np.zeros((1,D), dtype=np.float64)
		for j in idx:
			w = P[i,j]
			if i in cover_point_map[j] and w > 0.0:
				u, s, vt = np.linalg.svd(A_opt @ (A_opt.T @ phi(i, j)), full_matrices=False)
				i_idx = np.ravel(np.where(cover_point_map[j] == i))
				f_x = Fj[j][i_idx,:] + offsets[j]
				coords += w * (((A_opt.T @ u) @ vt) @ f_x.T).T
		assembly[i,:] = coords

	## Return the assembled coordinates 
	return(assembly)
	