# %% Alignment imports
import numpy as np
import numpy.typing as npt
from typing import Iterable, Dict
from itertools import combinations
from .sc import delta0D

# %% Alignment definitions
def global_translations(cover: Iterable, alignments: Dict):
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
def align_models(cover: Iterable, models: Dict):
	if len(cover) != len(models): raise ValueError("There should be a local euclidean model associated with each subset of the cover.")
	J = cover.index_set
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
	A, B = a - aC, b - bC
	
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



# %% 
# import matplotlib as mpl 
# import matplotlib.pyplot as pyplot 
# import pickle 
# flywing = pickle.load(open("flywing.pickle","rb"))
# pyplot.scatter(x=flywing[:,0], y=flywing[:,1], c=flywing[:,2])


# TODO: compare with 
# scipy.linalg.orthogonal_procrustes
      

# bv = BitVector.BitVector(size = 128)
# bv.count_bits()
# bv.next_set_bit(6)

# # %% 
# def binomial(n, r):
# 	''' Binomial coefficient, nCr, aka the "choose" function 
# 			n! / (r! * (n - r)!)
# 	'''
# 	p = 1    
# 	for i in range(1, min(r, n - r) + 1):
# 			p *= n
# 			p //= i
# 			n -= 1
# 	return p

# def nthresh(k, idx):
# 	"""Finds the largest value m such that C(m, k) <= idx."""
# 	mk = k
# 	while binomial(mk, k) <= idx:
# 		mk += 1
# 	return mk - 1


# def unrank_combn(rank, k):
# 	ret = []
# 	for i in range(k, 0, -1):
# 		element = nthresh(i, rank)
# 		ret.append(element)
# 		rank -= binomial(element, i)
# 	return ret

# # doesnt work 
# def rank_combn(input):
# 	ret = 0
# 	for k, ck in enumerate(sorted(input)):
# 		ret += binomial(ck, k + 1)
# 	return ret




# # %% 
# x = [unrank_combn(r, 2) for r in range(0, binomial(5,2)-1)]
# print(x)
# r = [rank_combn(combn) for combn in x]
# print(r)



# # %%
# m = np.random.randint(1,100,size=(100,100))
# m = np.tril(m) + np.triu(m.T)
# np.fill_diagonal(m, 0)
# print(m)
# print(floyd_warshall(m))

# # %%
# X = np.reshape([np.random.randn(8), np.random.randn(8)], (8, 2))
# pyplot.scatter(x=X[:,0], y=X[:,1])


# def enumerate(lazy, type=np.array):
# 	''' 
# 	Enumerates the values in (possibly nested) lazy generator object 'lazy', 
# 	coercing the result to a container type 'type' (which default to np.array)
# 	'''
# 	return(type([val for val in lazy]))


# def greedy_permutation(D):
#     """
#     A Naive O(N^2) algorithm to do furthest points sampling
    
#     Parameters
#     ----------
#     D : ndarray (N, N) 
#         An NxN distance matrix for points
#     Return
#     ------
#     tuple (list, list) 
#         (permutation (N-length array of indices), 
#         lambdas (N-length array of insertion radii))
#     """
    
#     N = D.shape[0]
#     # By default, takes the first point in the list to be the
#     # first point in the permutation, but could be random
#     perm = np.zeros(N, dtype=np.int64)
#     lambdas = np.zeros(N)
#     ds = D[0, :]
#     for i in range(1, N):
#         idx = np.argmax(ds)
#         perm[i] = idx
#         lambdas[i] = ds[idx]
#         ds = np.minimum(ds, D[idx, :])
#     return (perm, lambdas)

# # %% 
# def inverse_permutation(p):
# 	''' 
# 	Returns an array s, where s[i] gives the index of i in p.
# 	p is assumed to be a permutation of 0, 1, ..., len(p)-1
# 	'''
# 	s = np.empty_like(p)
# 	s[p] = np.arange(p.size)
# 	return(s)












# X = np.random.uniform(size=(50000,5))

# import time

# indices = np.random.choice(range(X.shape[0]), size = 500)

# Q = X[indices,:]
# t0 = time.time()
# linear_indices = np.zeros(500)
# for i in range(len(indices)):
# 	linear_indices[i] = np.where((X == Q[i,:]).all(axis=1))[0]
# t1 = time.time()
# total = t1-t0
# all(linear_indices == indices)

# t0 = time.time()
# log_indices = find_points(X[indices,:], X)
# t1 = time.time()
# total = t1-t0
# all(log_indices == indices)


# 0.04101300239562988

# x = X[501,:]

# find_points()




# X = np.reshape([np.random.uniform(size=50), np.random.uniform(size=50)], (50,2))
# lm = landmarks(X, 5)


# pyplot.draw()
# pyplot.scatter(x=X[:,0], y=X[:,1])
# pyplot.scatter(x=X[lm_idx,0], y=X[lm_idx,1], c="red")
# pyplot.show()

# # %% 
# # p = Point(138, 92)

# # for p in clarksongreedy.greedy(X, tree = True):
# # 	print(p)


# # help(clarksongreedy)
# # wut = clarksongreedy(X)
# # print(wut)
# # print(dir(greedypermutation))
# # print(p)



# # %%
# %%