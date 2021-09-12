# %% Alignment imports
import numpy as np
import numpy.typing as npt
from typing import Iterable, Dict
from itertools import combinations
from .sc import delta0D
from .cover import CoverLike

# %% Alignment definitions
def global_translations(cover: CoverLike, alignments: Dict):
	## Get optimal translation vectors 
	index_pairs = list(alignments.keys())
	d = len(alignments[index_pairs[0]]['translation'])
	rotations = { k: v['rotation'] for k,v in alignments.items() }
	translations = { k: v['translation'] for k,v in alignments.items() }
	
	## Evaluates pseudo-inverse on the coboundary matrix    
	J = len(cover)
	S = (np.fromiter(range(J), dtype=int), index_pairs) ## vertex/edge simplicial complex
	coboundary = delta0D(S, rotations)
	coboundary[np.isnan(coboundary)] = 0.0
	deltaX = np.linalg.pinv(coboundary)
	shiftE = np.zeros(d*len(S[1]))
	for (index,(i1,i2)) in enumerate(S[1]):
		shiftE[index*d:(index+1)*d] = translations[(i1,i2)]
	shiftV = np.matmul(deltaX,shiftE)

	## Offsets contain the translation vectors, keyed by edge indices in the nerve complex 
	offsets = { index : shiftV[index*d:(index+1)*d] for index in range(J) }
	return(offsets)

## Solve Procrustes problem for each non-empty intersection	
def align_models(cover: CoverLike, models: Dict):
	if len(cover) != len(models): raise ValueError("There should be a local euclidean model associated with each subset of the cover.")
	J = list(cover.keys())
	PA_map = {} # Procrustes analysis map
	for i, j in combinations(J, 2):
		subset_i, subset_j, ii, jj = cover[i], cover[j], J.index(i), J.index(j)
		ij_ind = np.intersect1d(subset_i, subset_j)
		if len(ij_ind) > 2:
			i_idx, j_idx = np.searchsorted(subset_i, ij_ind), np.searchsorted(subset_j, ij_ind) # assume subsets are ordered
			# PA_map[(ii,jj)] = old_procrustes(models[i][i_idx,:], models[j][j_idx,:], rotation_only=False, transform=False)
			PA_map[(ii,jj)] = opa(models[j][j_idx,:], models[i][i_idx,:], rotation_only=False, transform=False)
	return(PA_map)

def opa(a: npt.ArrayLike, b: npt.ArrayLike, transform=False, rotation_only=True):
	''' 
	Ordinary Procrustes Analysis:
	Determines the translation, orthogonal transformation, and uniform scaling of factor 
	that when applied to 'a' yields a point set that is as close to the points in 'b' under
	with respect to the sum of squared errors criterion.

	Example: 
		r,s,t,d = opa(a, b).values()
		
		## Rotate and scale a, then translate it => 'a' superimposed onto 'b'
		aligned_a = s * a @ r + t

	Returns:
		dictionary with rotation matrix R, relative scaling 's', and translation vector 't' 
		such that norm(b - (s * a @ r + t)) where norm(*) denotes the Frobenius norm.
	'''
	a, b = np.array(a, copy=False), np.array(b, copy=False)
	
	# Translation
	aC, bC = a.mean(0), b.mean(0) # centroids
	A, B = a - aC, b - bC         # center
	
	# Scaling 
	aS, bS = np.linalg.norm(A), np.linalg.norm(B)
	A /= aS 
	B /= bS

	# Rotation / Reflection
	U, Sigma, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
	R = U @ Vt
	
	# Correct to rotation if requested
	if rotation_only and np.linalg.det(R) < 0:
		d = np.sign(np.linalg.det(Vt.T @ U.T))
		Sigma = np.append(np.repeat(1.0, len(Sigma)-1), d)
		R = Vt.T @ np.diag(Sigma) @ U.T

	# Normalize scaling + translation 
	s = np.sum(Sigma) * (bS / aS)  	 # How big is B relative to A?
	t = bC - s * aC @ R              # place translation vector relative to B

	# Procrustes distance
	z = (s * a @ R + t)
	d = np.linalg.norm(z - b)**2
	
	# The transformed/superimposed coordinates
	# Note: (s*bS) * np.dot(B, aR) + c
	output = { "rotation": R, "scaling": s, "translation": t, "distance": d }
	if transform: output["coordinates"] = z
	return(output)
