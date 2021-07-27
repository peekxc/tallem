import numpy as np
import numpy.typing as npt
from typing import Callable, Iterable, List, Set, Dict, Optional, Tuple, Any, Union, Sequence
from itertools import combinations

from tallem import fast_svd
from tallem.sc import delta0D
from tallem.distance import dist
from tallem.cover import partition_of_unity
from tallem.samplers import uniform_sampler

import autograd.scipy.linalg as auto_scipy 
import autograd.numpy as auto_np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent


def _initial_frame(D, phi, n):
	Fb = np.vstack([phi(j).T for j in range(n)])
	Phi_N = Fb.T @ Fb
	Eval, Evec = np.linalg.eigh(Phi_N)
	return(Evec[:,np.argsort(-Eval)[:D]])

## Huber loss function to optimize
def huber_loss(embed_map: Callable, subsetter: Callable[[], Iterable[int]], epsilon: auto_np.float32 = 1.0, update_eps = lambda eps: eps):
	def cost_function(A):
		nonlocal epsilon 
		nuclear_norm = 0.0
		for j in subsetter():
			M = auto_np.array(A.T @ embed_map(j)) #  A.T @ phi(j), dtype='float')
			svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False)
			nuclear_norm += auto_np.sum([t if t >= epsilon else (t**2)/epsilon for t in auto_np.abs(svals)])
		epsilon = update_eps(epsilon)
		return(-nuclear_norm)
	return(cost_function)

## --- Optimization to find the best A matrix --- 
def frame_reduction(alignments: Dict, pou: npt.ArrayLike, D: int, fast_gradient = True):
	n, J, d = pou.shape[0], pou.shape[1], len(alignments[list(alignments.keys())[0]]['translation'])
	
	## Start off with StiefelLoss pybind11 module
	stf = fast_svd.StiefelLoss(n, d, D)

	## Initialize rotation matrix hashmap 
	I1 = [index[0] for index in alignments.keys()]
	I2 = [index[1] for index in alignments.keys()]
	R = np.vstack([pa['rotation'] for index, pa in alignments.items()])
	stf.init_rotations(I1, I2, R, J)

	## Populate frame matrix map
	for i in range(n): stf.populate_frame(i, np.ravel(pou[i,:].todense()), False)

	## Get the initial frame 
	Fb = stf.all_frames()
	Eval, Evec = np.linalg.eigh(Fb @ Fb.T)
	A0 = Evec[:,np.argsort(-Eval)[:D]]

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
	
	## Perform the optimization 
	Xopt = solver.solve(problem, x=A0)
	return(Xopt, stf)

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