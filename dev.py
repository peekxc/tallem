# %% Imports
import numpy as np 
import cppimport.import_hook
sys.path.insert(0, "src/tallem")

# %% Random data 
X = np.random.uniform(size=(10, 2))
y = np.random.uniform(size=(10, 1))

# %% Fast dense/sparse matrix multiplication 
# import carma_svd

# %% Test all the SVD implementations 
bn = example.BetaNuclearDense(1000, 3, 3)
bn.three_svd()



# %% 2 x 2 SVD
# Based on: https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
def svd_2x2(x):
	a,b,c,d = x[0,0], x[0,1], x[1,0], x[1,1]
	theta = 0.5*np.arctan2(2*a*c + 2*b*d, a**2 + b**2 - c**2 - d**2)
	phi = 0.5*np.arctan2(2*a*b + 2*c*d, a**2 - b**2 + c**2 - d**2) 
	ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
	u = np.array([[ct, -st],[st, ct]], dtype=np.float32)
	s1 = a**2 + b**2 + c**2 + d**2
	s2 = np.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4*(a*c + b*d)**2)
	s = np.array([ np.sqrt((s1 + s2)/2), np.sqrt((s1 - s2)/2) ])
	sign11 = 1 if ((a*ct+c*st)*cp + (b*ct + d*st)*sp) >= 0 else -1 
	sign22 = 1 if ((a*st-c*ct)*sp + (-b*st + d*ct)*cp) >= 0 else -1 
	v = np.array([[sign11*cp,-sign22*sp],[sign11*sp,sign22*cp]], dtype=np.float32)
	return((u,s,v))



y = np.hstack((x, np.zeros(shape=(3,1))))
print(np.linalg.svd(x))
print(np.linalg.svd(y))

# %% 2 x 2 SVD
import numpy as np
from tallem import fast_svd
x = np.random.uniform(size=(3,3))
y = fast_svd.fast_svd(x)


# %% Try import hook 
import numpy as np
x = np.random.uniform(size=(3,3))
import cppimport.import_hook
sys.path.insert(0, "src/tallem")
import fast_svd

fast_svd.fast_svd(x)

# %% carma 
import numpy as np
from tallem import carma_svd
X = np.random.uniform(size=(10,10))
y = np.random.uniform(size=(10,1))
print(carma_svd.ols(X, y))


# %% stiefel loss 
import numpy as np 
from tallem import fast_svd
d, D, J, n = 3, 3, 500, 10000
A = np.random.uniform(size=(d*J, D))
Phi = np.random.uniform(size=(d*J,d*n))

[A, R] = np.linalg.qr(A)

stf = fast_svd.StiefelLoss(n, d, D)
stf.output = A.T @ Phi

# %% stiefel gradient
%%time
f, gf = stf.gradient(Phi)

# %% stiefel gradient (3 x 3)
%%time
stf.three_svd()

# %% stiefel gradient carma (3 x 3)
%%time
stf.three_svd_carma()

# %% base python + numpy 
%%time
nuclear_norm = 0.0
gr = np.zeros((D, d))
for i in range(n):
	u,s,vt = np.linalg.svd(stf.output[:,(i*3):(i*3+3)], compute_uv=True, full_matrices=False)
	nuclear_norm += np.sum(np.abs(s))
	gr += u @ vt


# %% stiefel gradient
## Compare with python loop 



# %% testing tallem 
import numpy as np 
from tallem.datasets import mobius_band
from tallem.cover import IntervalCover
from tallem.cover import partition_of_unity

M, f = mobius_band(n_polar = 25, n_wide = 9, embed = 3).values()
B = f[:,1:2]
cover = IntervalCover(B, n_sets = 8, overlap = 0.25, gluing=[1])
PoU = partition_of_unity(B, cover, beta="triangular")

from tallem.mds import classical_MDS
from tallem.distance import dist
local_dim = 2
local_map = lambda x: classical_MDS(dist(x, as_matrix=True), k = local_dim)
local_models = { index : local_map(M[subset,:]) for index, subset in cover }

## Rotation, scaling, translation, and distance information for each intersecting cover subset
from tallem.procrustes import align_models
alignments = align_models(cover, local_models)


# %% Initialize Frame stuff
from tallem import fast_svd
d, D, J, n = local_dim, 3, len(cover), cover.n_points
stf = fast_svd.StiefelLoss(n, d, D)

## Initialize rotation matrix hashmap 
I1 = [index[0] for index in alignments.keys()]
I2 = [index[1] for index in alignments.keys()]
R = np.vstack([pa['rotation'] for index, pa in alignments.items()])
stf.init_rotations(I1, I2, R, J)

## Populate frame matrix map
for i in range(n):
	stf.populate_frame(i, np.ravel(PoU[i,:].todense()), False)

# %% Extract the frames
Fb = stf.all_frames()
Eval, Evec = np.linalg.eigh(Fb @ Fb.T)
A0 = Evec[:,np.argsort(-Eval)[:D]]

def stiefel_gradient(d_frame):
	stf.embed(d_frame.T) ## populates (D x dn) output array
	F, GF = stf.gradient()
	return(F / stf.n, GF / stf.n)

F, GF = stiefel_gradient(A0, stf)

for delta in range(10):
	loss, _ = stiefel_gradient(A0 - (delta/10)*GF, stf)
	print(loss)

from pymanopt.manifolds import Stiefel


Phi = stf.all_frames()
import autograd.numpy as auto_np
def sv_loss(A):
	out = A.T @ Phi
	val = 0.0
	for i in range(n):
		u,s,v = auto_np.linalg.svd(out[:,i*d:((i+1)*d)], full_matrices=False)
		val += auto_np.sum(auto_np.abs(s))
	return(-val/n)

from autograd import grad
gr = grad(sv_loss)
gr(A0)

from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
manifold = Stiefel(d*J, d)
problem = Problem(manifold=manifold, cost=sv_loss)
solver = SteepestDescent()
Xopt = solver.solve(problem, x=A0)

manifold.retr(X=A0, G=G.T)



def stiefel_loss(A):
	stf.embed(A.T) ## populates (D x dn) output array
	F, GF = stf.gradient()
	return(F / stf.n)
def stiefel_egrad(A):
	stf.embed(A.T) ## populates (D x dn) output array
	F, GF = stf.gradient()
	return(GF / stf.n)
manifold = Stiefel(d*J, d)
problem = Problem(manifold=manifold, cost=stiefel_loss, egrad=stiefel_egrad)
solver = SteepestDescent(mingradnorm=1e-12, maxiter=100, minstepsize=1e-14)
Xopt = solver.solve(problem, x=A0)


#-8.4632914625038969e-01	9.29798595e-07

from tallem.diagnostics import check_gradient
check_gradient(problem, x=A0)


# %% Test frame reduction (optimization)
from tallem.stiefel import frame_reduction
A, stf = frame_reduction(alignments, PoU, D, True)



# %% Run the assembly
## Construct the global assembly function 
assembly = np.zeros((n, D), dtype=np.float64)
coords = np.zeros((1,D), dtype=np.float64)
index_set = list(local_models.keys())
translations = global_translations(alignments)
for i in range(n):
	w_i = np.ravel(PoU[i,:].todense())
	nz_ind = np.where(w_i > 0)[0]
	d_frame = stf.get_frame(i) # already weighted 
	coords.fill(0)
	for j in nz_ind: 
		subset_j = cover[index_set[j]]
		relative_index = np.searchsorted(subset_j, i)
		if relative_index < len(subset_j) and subset_j[relative_index] == i:	
			# TODO: make dynamically weighted and determine whether it's bettier to precompute A @ A.T 
			u, s, vt = np.linalg.svd(A @ (A.T @ d_frame), full_matrices=False, compute_uv=True) 
			d_coords = local_models[index_set[j]][relative_index,:]
			coords += w_i[j]*A.T @ (u @ vt) @ (d_coords + translations[j])
	assembly[i,:] = coords



# %% Procrustes 
import matplotlib.pyplot as plt
star1 = np.array([[131,38], [303, 39], [357, 204], [217,305], [78,204]])
star2 = np.array([[61, 33],[120, 73],[103,162],[32,177],[7,97]])
plt.scatter(star1[:,0], star1[:,1], c="red")
plt.scatter(star2[:,0], star2[:,1], c="blue")

from tallem.procrustes import ord_procrustes
pa = ord_procrustes(star1, star2)

plt.scatter(A[:,0], A[:,1])
plt.scatter(B[:,0], B[:,1])

r,s,t,d = ord_procrustes(star1, star2).values()

plt.scatter(star1[:,0], star1[:,1], c="red")
#Star2 = (pa['rotation'] @ star2.T).T + pa['translation']
Star2 = s * star2 @ r + t
plt.scatter(Star2[:,0], Star2[:,1], c="blue")

plt.scatter(star2[:,0], star2[:,1], c="blue")
#Star1 = ((star1 * 1/pa['scaling']) @ pa['rotation'].T) + pa['translation']
plt.scatter(Star1[:,0], Star1[:,1], c="red")

r,s,t,d = ord_procrustes(a, b).values()
plt.scatter(a[:,0], a[:,1], c="red")
plt.scatter(b[:,0], b[:,1], c="blue")

Z = s*a@r+t 
plt.scatter(Z[:,0], Z[:,1], c="orange")

