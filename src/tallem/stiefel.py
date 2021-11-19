import numpy as np
import numpy.typing as npt
from typing import *
from itertools import combinations
from scipy.sparse import csc_matrix, issparse

from .sc import delta0D
from .distance import dist
from .cover import partition_of_unity
from .samplers import uniform_sampler
from .utility import * 

## Import the specific extension module we need
from tallem.extensions import fast_svd

def _initial_frame(D, phi, n):
	Fb = np.vstack([phi(j).T for j in range(n)])
	Phi_N = Fb.T @ Fb
	Eval, Evec = np.linalg.eigh(Phi_N)
	return(Evec[:,np.argsort(-Eval)[:D]])

## Huber loss function to optimize
# def huber_loss(embed_map: Callable, subsetter: Callable[[], Iterable[int]], epsilon: auto_np.float32 = 1.0, update_eps = lambda eps: eps):
# 	def cost_function(A):
# 		nonlocal epsilon 
# 		nuclear_norm = 0.0
# 		for j in subsetter():
# 			M = auto_np.array(A.T @ embed_map(j)) #  A.T @ phi(j), dtype='float')
# 			svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False)
# 			nuclear_norm += auto_np.sum([t if t >= epsilon else (t**2)/epsilon for t in auto_np.abs(svals)])
# 		epsilon = update_eps(epsilon)
# 		return(-nuclear_norm)
# 	return(cost_function)


## --- Optimization to find the best A matrix --- 
def frame_reduction(alignments: Dict, pou: csc_matrix, D: int, optimize=False, fast_gradient=False, **kwargs):
	assert isinstance(pou, csc_matrix), "Partition of unity must be represented as a CSC sparse matrix"
	n, J, d = pou.shape[0], pou.shape[1], len(alignments[list(alignments.keys())[0]]['translation'])
	
	## Start off with StiefelLoss pybind11 module
	stf = fast_svd.StiefelLoss(n, d, D)

	## Initialize rotation matrix hashmap 
	I1 = [index[0] for index in alignments.keys()]
	I2 = [index[1] for index in alignments.keys()]
	
	# Stack Omegas contiguously
	# R is (dL x d), where 0 <= L <= (J choose 2) (num. non-empty intersections between cover sets)
	R = np.vstack([pa['rotation'] for index, pa in alignments.items()]) 
	stf.init_rotations(I1, I2, R, J)

	# ## Populate frame matrix map
	# iota = np.array(pou.argmax(axis=1)).flatten()
	# pou_t = pou.transpose().tocsc()
	# stf.populate_frames(iota, pou_t, False) # populate all the iota-mapped frames in vectorized fashion

	# ## Get the initial frame 
	# Fb = stf.all_frames() ## Note these are already weighted w/ the sqrt(varphi)'s!
	# Eval, Evec = np.linalg.eigh(Fb @ Fb.T)
	# A0 = Evec[:,np.argsort(-Eval)[:D]]

	## Store partition of unity in CSC matrix internally to prepare for frame population
	stf.setup_pou(pou.transpose().tocsc())
	iota = np.ravel(stf.extract_iota())
	
	## Use the bijection given by iota to generate the (sparse) Phi matrix 
	stf.populate_frames_sparse(iota)

	## Compute the initial guess, return it as optimal if not optimizing further
	ew, A0 = stf.initial_guess(D, True)
	if (not(optimize)): 
		return(A0, A0, stf)
	else:
		ask_package_install("autograd")
		ask_package_install("pymanopt")
		import autograd.scipy.linalg as auto_scipy 
		import autograd.numpy as auto_np
		from pymanopt.manifolds import Stiefel
		from pymanopt import Problem
		from pymanopt.solvers import SteepestDescent
		
		## Setup optimization using Pymanopt
		manifold = Stiefel(d*J, D)

		solver = SteepestDescent(mingradnorm=1e-12, maxiter=100, minstepsize=1e-14)
		if (fast_gradient):
			stiefel_cost = lambda A: -stf.gradient(A.T, True)[0]
			stiefel_gradient = lambda A: -stf.gradient(A.T, True)[1]
			problem = Problem(manifold=manifold, cost=stiefel_cost, egrad=stiefel_gradient)
		else: 
			sampler = uniform_sampler(n)
			problem = Problem(manifold=manifold, cost=huber_loss(lambda i: stf.get_frame(i), lambda: sampler(n), 0.30))
			Xopt = solver.solve(problem, x=A0)
		return(A0, Xopt, stf)
	
	# Need Stiefel(n=d*J,p=D) as Stiefel(n,p) := space of (n x p) orthonormal matrices


	# ## Huber loss function to optimize
	# def huber_loss(subsetter: Callable[[], Iterable[int]], epsilon: auto_np.float32 = 1.0, update_eps = lambda eps: eps):
	# 	def cost_function(A):
	# 		nonlocal epsilon 
	# 		nuclear_norm = 0.0
	# 		for j in subsetter():
	# 			M = auto_np.array(A.T @ phi(j), dtype='float')
	# 			svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False)
	# 			nuclear_norm += auto_np.sum([t if t >= epsilon else (t**2)/epsilon for t in auto_np.abs(svals)])
	# 		epsilon = update_eps(epsilon)
	# 		return(-nuclear_norm)
	# 	return(cost_function)