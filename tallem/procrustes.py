# %% Procrustes imports
import numpy as np
import numpy.typing as npt

# %% Procrustes definitions
def ord_procrustes(a: npt.ArrayLike, b: npt.ArrayLike, transform=False, rotation_only=False):
	''' 
	Ordinary Procrustes Analysis:
	Determines the translation, orthogonal transformation, and uniform scaling of factor 
	that when applied to 'b' yields a point set that is as close to the points in 'a' under
	with respect to the sum of squared errors criterion.

	Returns:
		dictionary with rotation matrix R, relative scaling 's', and translation vector 't' 
		such that a \approx (s*norm(b)) * b@R + t where norm(*) denotes the Frobenius norm.
	'''
	a, b = np.array(a), np.array(b)
	
	# Translation
	aT, bT = a.mean(0), b.mean(0)
	A, B = a - aT, b - bT
	
	# Scaling 
	aS, bS = np.linalg.norm(A), np.linalg.norm(B)
	A /= aS 
	B /= bS

	# Rotation / Reflection
	U, Sigma, Vt = np.linalg.svd(np.dot(B.T, A), full_matrices=False)
	aR = np.dot(U, Vt)
	
	# Correct to rotation if requested
	if rotation_only and np.linalg.det(aR) < 0:
		Vt[:,-1] *= -1 # note that Sigma is changing implicitly here
		Sigma = np.repeat(1.0, len(Sigma))
		Sigma[-1] = -1.0
		aR = np.dot(np.dot(U, np.diag(Sigma)), Vt)

	# Normalize scaling + translation 
	s = np.sum(Sigma) * (aS / bS)  	 # How big is A relative to B?
	c = aT - s * np.dot(bT, aR) # place translation vector relative to A

	# Procrustes distance
	# aD = np.sqrt(np.sum((A - B.dot(aR))**2) / len(a))
	D = 1.0 - np.sum(Sigma)**2
	
	# The transformed/superimposed coordinates
	# Note: (s*bS) * np.dot(B, aR) + c
	output = { "rotation": aR, "scaling": s, "translation": c, "distance": D }
	if transform:
		Z = (s*bS) * np.dot(B, aR) + c
		output["coordinates"] = Z
	return(output)


# %% Write the flywing data set to disk
# import numpy as np
# import pickle
# arr1 = np.array([[588.0, 443.0], [178.0, 443.0], [56.0, 436.0], [50.0, 376.0], [129.0, 360.0], [15.0, 342.0], [92.0, 293.0], [79.0, 269.0], [276.0, 295.0], [281.0, 331.0], [785.0, 260.0], [754.0, 174.0], [405.0, 233.0], [386.0, 167.0], [466.0, 59.0]])
# arr2 = np.array([[477.0, 557.0], [130.129, 374.307], [52.0, 334.0], [67.662, 306.953], [111.916, 323.0], [55.119, 275.854], [107.935, 277.723], [101.899, 259.73], [175.0, 329.0], [171.0, 345.0], [589.0, 527.0], [591.0, 468.0], [299.0, 363.0], [306.0, 317.0], [406.0, 288.0]])
# label1 = np.repeat(1, arr1.shape[0]).reshape((arr1.shape[0], 1))
# label2 = np.repeat(2, arr2.shape[0]).reshape((arr2.shape[0], 1))
# flywing = np.vstack((
# 	np.hstack((arr1, label1)),
# 	np.hstack((arr2, label2))
# ))
# with open('flywing.pickle', 'wb') as output:
#     pickle.dump(flywing, output)

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
