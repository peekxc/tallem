# %% 
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))	
	
# %% Setup 
import numpy as np
from src.tallem.datasets import mobius_band
from src.tallem.cover import IntervalCover
from src.tallem import TALLEM

M = mobius_band(120, 15, plot=False, embed=6)
X, B = M['points'], M['parameters'][:,[1]]

## Run TALLEM on polar coordinate cover 
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(B, n_sets = 30, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)
top = TALLEM(cover, local_map="pca2", n_components=3)
embedding = top.fit_transform(X, B)


# %% Attempt #1: generate frames Phi_x in python, on demand
n, d, D, J = top._stf.n, top._stf.d, top._stf.D, len(top.cover)
P = top.pou 
A = top.alignments
def phi(i, j = None):
	J = P.shape[1]
	k = np.argmax(P[i,:]) if j is None else j
	def weighted_omega(j):
		nonlocal i, k
		w = np.sqrt(P[i,j])
		pair_exists = np.array([pair in A.keys() for pair in [(k,j), (j,k)]])
		if w == 0.0 or not(pair_exists.any()):
			return(w*np.eye(d))
		return(w*A[(k,j)]['rotation'] if pair_exists[0] else w*A[(j,k)]['rotation'].T)
	return(np.vstack([weighted_omega(j) for j in range(J)]))

# %% Regular Python generation 
%%time
Phi = np.zeros(shape=(d*J, d*n))
for i in range(n):
	Phi[:,(i*d):((i+1)*d)] = phi(i)
# ~ 11.5 s

# %% Setup numba 
from numba import jit, njit, prange
from numba.core import types
from numba.typed import Dict
import numpy as np

iota = np.ravel(top.pou.argmax(axis=1).flatten())
pou = top.pou
n, d, D, J = top._stf.n, top._stf.d, top._stf.D, len(top.cover)
A = top.alignments

@jit(forceobj=True, parallel=True) 
def phi(i): ## assume iota + constants d, J, and n are defined
	out = np.zeros(shape=(d*J, d))
	for j in prange(J):
		k, w = iota[i], np.sqrt(pou[i,j])
		# key = rank_comb2(i=k,j=j,n=J)
		if w == 0.0 or not((j,k) in A.keys()) and not((k,j) in A.keys()):
			out[(j*d):((j+1)*d),:] = w*np.eye(d)
		else: 
			out[(j*d):((j+1)*d),:] = w*(A[(k,j)]['rotation']) if k < j else A[(j,k)]['rotation'].T
	return(out)

# %% Numba attempt 1
%%time
Phi = np.zeros(shape=(d*J, d*n))
for i in range(n): Phi[:,(i*d):((i+1)*d)] = phi(i)
# ~ 7.26 s


# %% Numba attempt 2 O(nJ + |R|  + dJ*dn) memory 
## Automatically monotonically increasing
rank2 = lambda i, j, n: np.int32(n*i - i*(i+1)/2 + j - i - 1) if i < j else np.int32(n*j - j*(j+1)/2 + i - j - 1)
pairs = np.array([rank2(i=i,j=j,n=J) for i,j in A.keys()], dtype=np.int32)
n_pairs = len(pairs)

## Make contiguous dense numpy matrices
R = np.vstack([A[index]['rotation'] for index in A.keys()])
P_dense = top.pou.A

@jit(nopython=True, parallel=True) 
def populate_frames():
	Phi = np.zeros(shape=(d*J, d*n))
	for i in prange(n):
		for j in prange(J):
			k, w = iota[i], np.sqrt(P_dense[i,j])
			if (w == 0.0): continue
			key = np.int32(n*k - k*(k+1)/2 + j - k - 1) if k < j else np.int32(n*j - j*(j+1)/2 + k - j - 1)
			ind = np.searchsorted(pairs, key)
			if ((ind < n_pairs) and (pairs[ind] == key)):
				if j < k:
					Phi[(j*d):((j+1)*d),(i*d):((i+1)*d)] = w*R[(ind*d):((ind+1)*d),:]
				else:
					Phi[(j*d):((j+1)*d),(i*d):((i+1)*d)] = w*R[(ind*d):((ind+1)*d),:].T
			else:
				Phi[(j*d):((j+1)*d),(i*d):((i+1)*d)] = w*np.eye(d)
	return(Phi)
		
# %% Numba attempt 2 
%%time 
Phi = populate_frames() # ~ 2 ms 
# ~ 3.03 ms
# 22954

# %% Numba attempt 3 (setup) --- O(nJ + n + J^2 * d^2  + dJ*dn) memory 
keys = [(i,j) for i in range(J) for j in range(J)]
R = np.vstack([A[key]['rotation'] if key in A.keys() else np.eye(d) for key in keys])

@jit(nopython=True, parallel=True) 
def populate_frames():
	Phi = np.zeros(shape=(d*J, d*n))
	for i in prange(n):
		for j in prange(J):
			k, w = iota[i], np.sqrt(P_dense[i,j])
			key = k + J*j
			Phi[(j*d):((j+1)*d),(i*d):((i+1)*d)] = w*R[(key*d):((key+1)*d),:]
	return(Phi)

# %% Numba attempt 3
%%time 
Phi = populate_frames() # 4.83 ms
# 27576

# %% C++ attempt 1 (setup)
P = top.pou.transpose().tocsc()
iota = np.ravel(top.pou.argmax(axis=1).flatten())

# %% C++ attempt 1 
%%time
top._stf.populate_frames(iota, P, False) 
# 32.8 ms


# %% C++ attempt 1 (sparse) 
%%time
top._stf.populate_frames(iota, P, True)


# %% C++ version attempt 2 (sparse)
%%time
top._stf.populate_frames_sparse(P) # ~ 18 ms
# 25.4 ms
# len(top._stf.all_frames_sparse()[2])
# 32088

# %% 


	# ## Start off with StiefelLoss pybind11 module
	# stf = fast_svd.StiefelLoss(n, d, D)

	# ## Initialize rotation matrix hashmap 

	
	# # Stack Omegas contiguously
	# I1 = [index[0] for index in alignments.keys()]
	# I2 = [index[1] for index in alignments.keys()]
	# R = np.vstack([pa['rotation'] for index, pa in alignments.items()]) 
	# stf.init_rotations(I1, I2, R, J)




	## Populate frame matrix map
	# iota = np.array(pou.argmax(axis=1)).flatten()
	# pou_t = pou.transpose().tocsc()
	# stf.populate_frames(I, pou_t, False) # populate all the iota-mapped frames in vectorized fashion

	# ## Get the initial frame 
	# Fb = stf.all_frames() ## Note these are already weighted w/ the sqrt(varphi)'s!
	# Eval, Evec = np.linalg.eigh(Fb @ Fb.T)
	# A0 = Evec[:,np.argsort(-Eval)[:D]]

	## Compute the initial guess
	stf.populate_frames_sparse(pou.transpose().tocsc()) # populate all the iota-mapped frames in vectorized fashion



# ## Automatically monotonically increasing
# pairs = np.array([], dtype=np.int32)

# ## Make a typed dictionary to enable Numba support for readingi
# R_dict = Dict.empty(
# 	key_type=types.int32,
#   value_type=types.float64[:,::1]
# )
# for i,j in A.keys():
# 	key = np.int32(rank_comb2(i,j,J))
# 	omega = np.array(A[(i,j)]['rotation'], dtype=np.float64)
# 	R_dict[key] = omega