import numpy as np
import numpy.typing as npt
from typing import Callable, Iterable, List, Set, Dict, Optional, Tuple, Any, Union, Sequence
from itertools import combinations
from tallem.sc import delta0D
from tallem.distance import dist
from tallem.cover import partition_of_unity
from tallem.mds import classical_MDS
from tallem.procrustes import ord_procrustes
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

## --- Optimization to find the best A matrix --- 
def frame_reduction():
	
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
	A0 = _initial_frame(D, phi, X.shape[0])
	Xopt = solver.solve(problem=problem, x=A0)