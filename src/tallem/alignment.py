# %% Alignment imports
import numpy as np
import numpy.typing as npt
from typing import Iterable, Dict
from itertools import combinations
from scipy.linalg import orthogonal_procrustes
from .sc import delta0D
from .cover import CoverLike

# %% Alignment definitions
# This was taken from: https://gitlab.msu.edu/mikejosh/tallm/-/tree/master/PyTALLEM
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
def align_models(cover: CoverLike, models: Dict, **kwargs):
	if len(cover) != len(models): raise ValueError("There should be a local euclidean model associated with each subset of the cover.")
	index_set = list(cover.keys())
	PA_map = {} # Procrustes analysis map
	for i, j in combinations(index_set, 2):
		subset_i, subset_j, ii, jj = cover[i], cover[j], index_set.index(i), index_set.index(j)
		ij_ind, i_idx, j_idx = np.intersect1d(subset_i, subset_j, return_indices=True)
		if len(ij_ind) > 2:
			PA_map[(ii,jj)] = opa(models[j][j_idx,:], models[i][i_idx,:], **kwargs)
	return(PA_map)

# def opa(a: npt.ArrayLike, b: npt.ArrayLike, scale=True, coords=False, fit="best"):
# 	''' 
# 	Ordinary Procrustes Analysis:
# 	Determines the translation, orthogonal transformation, and uniform scaling of factor 
# 	that when applied to 'a' yields a point set that is as close to the points in 'b' under
# 	with respect to the sum of squared errors criterion.

# 	Example: 
# 		R,s,t,d = opa(a, b).values()
		
# 		## Rotate, scale, and then translate 'a' to superimpose onto 'b'
# 		aligned_a = scale_points(a, s) @ R + t = scale_points(a @ r, s) + t

# 	Returns:
# 		dictionary with rotation matrix R, relative scaling 's', and translation vector 't' 
# 		such that norm(b - (s * a @ r + t)) where norm(*) denotes the Frobenius norm.
# 	'''
# 	a, b = np.array(a, copy=False), np.array(b, copy=False)
# 	assert np.all(a.shape == b.shape)
# 	dim = a.shape[1]
	
# 	# Translation
# 	aC, bC = a.mean(0), b.mean(0) # centroids
# 	a, b = a - aC, b - bC         # center
	
# 	# Scaling 
# 	aS, bS = 1.0, 1.0 
# 	if scale:
# 		aS, bS = np.linalg.norm(a), np.linalg.norm(b)
# 		a /= aS 
# 		b /= bS

# 	# Normalize scaling + translation relative to A
# 	s = (bS / aS)  	# How big is B relative to A?
# 	t = bC - aC     # How to translate A to B?  

# 	# Get the orthogonal rotation/reflection/rotoreflection
# 	U, svals, Vt = np.linalg.svd(b.T @ a)
# 	R = U @ Vt # R, ss = orthogonal_procrustes(a, b)

# 	## Also compute rotation matrix
# 	d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
# 	S = np.diag([1] * (dim - 1) + [d])
# 	R_rot = U @ S @ Vt

# 	## Choose the rotation matrix
# 	if fit == "best":
# 		d_any = np.linalg.norm(a @ R - b, "fro")
# 		d_rot = np.linalg.norm(a @ R_rot - b, "fro")
# 		if d_rot < d_any or abs(d_rot - d_any) <= 1e-8:
# 			R = R_rot # prefer rotation matrices if the fit is the same or better
# 	elif fit == "rotation-only":
# 		R = R_rot
# 	else: 
# 		raise ValueError("Invalid fit parameter")

# 	# Procrustes error
# 	# NOTE: at this point, a and b may be re-scaled and centered 
# 	p_error = np.linalg.norm((R @ a.T - b.T).T, "fro")
	
# 	# Output either the operations or the transformed/superimposed coordinates
# 	if coords:
# 		A = (s * (R @ (a*aS).T)).T 
# 		return(A+aC+t)
# 	else: 
# 		output = { "rotation": R, "scaling": s, "translation": t, "distance": p_error }
# 		return(output)


# def opa(a: npt.ArrayLike, b: npt.ArrayLike, coords=False, scale = False, fit="best"):
# 	''' 
# 	Ordinary Procrustes Analysis:
# 	Determines the translation, orthogonal transformation, and uniform scaling of factor 
# 	that when applied to 'a' yields a point set that is as close to the points in 'b' under
# 	with respect to the sum of squared errors criterion.
# 	Example: 
# 		r,s,t,d = opa(a, b).values()
		
# 		## Rotate and scale a, then translate it => 'a' superimposed onto 'b'
# 		aligned_a = s * a @ r + t
# 	Returns:
# 		dictionary with rotation matrix R, relative scaling 's', and translation vector 't' 
# 		such that norm(b - (s * a @ r + t)) where norm(*) denotes the Frobenius norm.
# 	'''
# 	a, b = np.array(a, copy=False), np.array(b, copy=False)
	
# 	# Translation
# 	aC, bC = a.mean(0), b.mean(0) # centroids
# 	A, B = a - aC, b - bC         # center
	
# 	# Scaling 
# 	aS, bS = np.linalg.norm(A), np.linalg.norm(B)
# 	A /= aS 
# 	B /= bS

# 	# Rotation / Reflection
# 	U, Sigma, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
# 	R = U @ Vt
	
# 	# Correct to rotation if requested
# 	if np.linalg.det(R) < 0:
# 		d = np.sign(np.linalg.det(Vt.T @ U.T))
# 		Sigma = np.append(np.repeat(1.0, len(Sigma)-1), d)
# 		R = Vt.T @ np.diag(Sigma) @ U.T

# 	# Normalize scaling + translation 
# 	s = np.sum(Sigma) * (bS / aS)  	 # How big is B relative to A?
# 	t = bC - s * aC @ R              # place translation vector relative to B

# 	# Procrustes distance
# 	# z = (s * a @ R + t)
# 	# d = np.linalg.norm(z - b)**2
# 	d = np.linalg.norm(A @ R - B, "fro")
	
# 	# The transformed/superimposed coordinates
# 	# Note: (s*bS) * np.dot(B, aR) + c
# 	output = { "rotation": R, "scaling": s, "translation": t, "distance": d }
# 	if coords: return(s * a @ R + t)
# 	else:
# 		return(output)




def opa(X, Y, coords=False, scale=True, fit='best'):
	"""
	A port of MATLAB's `procrustes` function to Numpy.

	Procrustes analysis determines a linear transformation (translation,
	reflection, orthogonal rotation and scaling) of the points in Y to best
	conform them to the points in matrix X, using the sum of squared errors
	as the goodness of fit criterion.

			d, Z, [tform] = procrustes(X, Y)

	Inputs:
	------------
	X, Y    
			matrices of target and input coordinates. they must have equal
			numbers of  points (rows), but Y may have fewer dimensions
			(columns) than X.

	scaling 
			if False, the scaling component of the transformation is forced
			to 1

	reflection
			if 'best' (default), the transformation solution may or may not
			include a reflection component, depending on which fits the data
			best. setting reflection to True or False forces a solution with
			reflection or no reflection respectively.

	Outputs
	------------
	d       
			the residual sum of squared errors, normalized according to a
			measure of the scale of X, ((X - X.mean(0))**2).sum()

	Z
			the matrix of transformed Y-values

	tform   
			a dict specifying the rotation, translation and scaling that
			maps X --> Y

	"""
	X,Y = Y,X
	n,m = X.shape
	ny,my = Y.shape

	muX = X.mean(0)
	muY = Y.mean(0)

	X0 = X - muX
	Y0 = Y - muY

	ssX = (X0**2.).sum()
	ssY = (Y0**2.).sum()

	# centred Frobenius norm
	normX = np.sqrt(ssX)
	normY = np.sqrt(ssY)

	# scale to equal (unit) norm
	X0 /= normX
	Y0 /= normY

	if my < m:
			Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

	# optimum rotation matrix of Y
	A = np.dot(X0.T, Y0)
	U,s,Vt = np.linalg.svd(A,full_matrices=False)
	V = Vt.T
	T = np.dot(V, U.T)

	# # does the current solution use a reflection?
	# have_reflection = np.linalg.det(T) < 0

	# if that's not what was specified, force another reflection
	# if have_reflection:
	# 		V[:,-1] *= -1
	# 		s[-1] *= -1
	# 		T = np.dot(V, U.T)

	traceTA = s.sum()

	if scale:

			# optimum scaling of Y
			b = traceTA * normX / normY

			# standarised distance between X and b*Y*T + c
			d = 1 - traceTA**2

			# transformed coords
			Z = normX*traceTA*np.dot(Y0, T) + muX

	else:
			b = 1
			d = 1 + ssY/ssX - 2 * traceTA * normY / normX
			Z = normY*np.dot(Y0, T) + muX

	# transformation matrix
	if my < m:
			T = T[:my,:]
	c = muX - b*np.dot(muY, T)
	
	#transformation values 
	tform = {'rotation':T, 'scaling':b, 'translation':c, 'distance': d}
	return tform

