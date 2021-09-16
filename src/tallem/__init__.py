## TALLEM __init__.py file 
## Imports all the main utilities of TALLEM and provides a wrapper for the entire function

## Base external imports 
import re
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import Callable, Iterable, List, Set, Dict, Optional, Tuple, Any, Union, Sequence
from itertools import combinations
from scipy.sparse import issparse, csc_matrix

# TODO: define __all__
# __all__ = [
# 	'function_1',
# 	'function_2'
# ]

## tallem-specific relative imports
from .sc import delta0D
from .distance import dist
from .alignment import opa, align_models, global_translations
from .samplers import uniform_sampler
from .stiefel import frame_reduction
from .cover import IntervalCover, partition_of_unity
from .utility import find_where
from .assembly import assemble_frames, assembly_fast
from .cover import CoverLike
from .dimred import *

## To eventually remove or make optional to install
import autograd.scipy.linalg as auto_scipy 
import autograd.numpy as auto_np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

class TALLEM():
	'''
	TALLEM: Topological Assembly of Locally Euclidean Models

	__init__(cover, local_map, n_components, pou) -> (TALLEM instance): 
		cover := CoverLike iterable of length J that covers some topological space B, where B is the image of some map f : X -> B
		local_map := a callable mapping (m x p) subsets of X to some (m x d) space, where d < p, which approximately preserves the metric on X. 
		n_components := target embedding dimension
		pou := partition of unity, either one of ['triangular', 'quadratic'], or an (n x J) ArrayLike object whose rows
					 yield weights indicating the strength of membership of that point with each set in the cover.

	fit(X, B, pou) -> (TALLEM instance): 
		X := an (n x p) numpy array representing *n* points in *p* space.
		B := an (n x q) numpy array representing the image of f : X -> B, where f is a map that cpatures the topology and non-linearity of X. 

	fit_transform() -> ArrayLike:

	transform() -> ArrayLike:

	'''
	
	def __init__(self, cover: CoverLike, local_map: Union[str, Callable[npt.ArrayLike, npt.ArrayLike]], n_components: int = 2, pou: Union[str, csc_matrix] = "triangular"):
		assert isinstance(cover, CoverLike), "cover argument must be an CoverLike."
		
		## Store callable as part of the cover's initialization
		self.D = n_components ## Target dimension of the embedding (D) 

		## If string supplied to shortcut dim. reduction step, parse that, otherwise expect it to be a callable
		if isinstance(local_map, str):
			dr_methods = ["mmds", "cmds", "nmds", "pca", "iso", "lle", "le", "hlle"]
			method, d = re.findall(r'(\w+)(\d+)', local_map)[0]
			assert method in dr_methods, "Invalid method supplied"
			def dr_function(f, d): 
				return(lambda x: f(x, d))
			local_f = [mmds, cmds, nmds, pca, isomap][dr_methods.index(method)]
			self.local_map = dr_function(local_f, int(d))
		else:
			assert isinstance(local_map, Callable), "local model map must be a function."
			self.local_map = local_map
		
		## Assign cover as-is, validate on fit
		self.cover = cover 

		## Validate the partition of unity respects the closure condition relative to the cover 
		if isinstance(pou, csc_matrix): assert pou.shape[1] == len(self.cover), "Partition of unity must have one column per element of the cover"
		self.pou = pou

	def fit(self, X: npt.ArrayLike, B: Optional[npt.ArrayLike] = None, pou: Optional[Union[str, csc_matrix]] = None, **kwargs):
		X = np.asanyarray(X) 
		if B is None: B = X
		if X.shape[0] != B.shape[0]: raise ValueError("X and B must have the same number of rows.")
		if X.ndim == 1: X = X.reshape((len(X), 1))
		if B.ndim == 1: B = B.reshape((len(B), 1))
		if pou is None: pou = self.pou
		
		## Validate cover 
		membership = np.zeros(X.shape[0], dtype=bool)
		for ind in self.cover.values(): membership[ind] = True
		assert np.all(membership == True), "Supplied cover invalid: the union of the values does not contain all of B as a subset."
		
		## Map the local euclidean models
		## Note: the extra array constructor ensures singleton subsets are reported as matrices
		self.models = { index : self.local_map(X[np.array(subset),:]) for index, subset in self.cover.items() }

		## Construct a partition of unity
		J = len(self.cover)
		if isinstance(pou, str):
			## In this case, cover must have a set_distance(...) function!
			## This is where the coordinates of B are needed!
			self.pou = partition_of_unity(B, cover = self.cover, similarity = pou) 
		elif issparse(pou): 
			if pou.shape[1] != len(self.cover):
				raise ValueError("Partition of unity must have one column per element of the cover")
			for j, index in enumerate(self.cover.keys()):
				pou_nonzero = np.where(pou[:,j].todense() > 0)[0]
				is_invalid_pou = np.any(find_where(pou_nonzero, self.cover[index]) is None)
				if (is_invalid_pou):
					raise ValueError("The partition of unity must be supported on the closure of the cover elements.")
		else: 
			raise ValueError("Invalid partition of unity supplied. Must be either a string or a csc_matrix")

		## Align the local reference frames using Procrustes
		self.alignments = align_models(self.cover, self.models)
		
		## Get global translation vectors using cocyle condition
		self.translations = global_translations(self.cover, self.alignments)

		## Solve the Stiefel manifold optimization for the projection matrix 
		self.A0, self.A, self._stf = frame_reduction(self.alignments, self.pou, self.D, **kwargs)

		## Assemble the frames!
		## See: https://github.com/rasbt/python-machine-learning-book/blob/master/faq/underscore-convention.md
		self.embedding_ = assembly_fast(self._stf, self.A, self.cover, self.pou, self.models, self.translations)
		return(self)

	def fit_transform(self, X: npt.ArrayLike, B: Optional[npt.ArrayLike] = None, **fit_params) -> npt.ArrayLike:		
		self.fit(X, B, **fit_params)
		return(self.embedding_)

	def __repr__(self) -> str:
		return("TALLEM instance")

	def assemble(self, pou: Optional[csc_matrix] = None, D_frame: Optional[npt.ArrayLike] = None) -> npt.ArrayLike:
		if D_frame is None: D_frame = self.A
		if pou is None: pou = self.pou
		return(assembly_fast(self._stf, D_frame, self.cover, pou, self.models, self.translations))

	def _profile(self, **kwargs):
		import line_profiler
		profile = line_profiler.LineProfiler()
		profile.add_function(self.fit)
		profile.add_function(partition_of_unity)
		profile.add_function(frame_reduction)
		profile.add_function(assembly_fast)
		profile.enable_by_count()
		self.fit(**kwargs)
		profile.print_stats(output_unit=1e-3)
	