# %% Imports + configurations
import numpy as np 
from tallem import tallem_transform
from tallem.datasets import mobius_band
sys.path.insert(0, "src/tallem")
np.set_printoptions(linewidth=300)

# %% Generate dataset 
M = mobius_band(embed=6)
X = M['points']
f = M['parameters'][:,0]

# %% Run tallem
%%time
Y = tallem_transform(X, f, D=3, J=10)
## Show profile_tallem.py

# %% Equation 9 - Matrix multiplication
# ( Show times in tallem_testing.py )

# %% Equation 9 - SVD
from scipy.sparse import random as sparse_random
import numpy as np 

from numpy.linalg import svd										 # LAPACK - uses divide and conquer
from scipy.linalg import svdvals 								 # only compute singular values
from scipy.linalg import svd as scipy_svd				 # LAPACK, but has option for gesdd or gesvd, lapack_driver='gesdd'
from sklearn.utils.extmath import randomized_svd # Randomized 
from jax.numpy.linalg import svd as jax_svd			 # uses LAX backend 
from tallem import example                       # Disney svd 

# %% SVD benchmarks
import timeit
for nr in [20, 100, 200]:
	for density in [0.01, 0.05, 0.25, 0.75]:
		X = sparse_random(nr, nr, density=density, format='csr')
		X_np = X.toarray()
		run_sec_random = timeit.repeat(lambda: randomized_svd(X, n_components=2), number=50)
		run_sec_np = timeit.repeat(lambda: np.linalg.svd(X_np, full_matrices=True, compute_uv=True), number=50)
		run_sec_jax = timeit.repeat(lambda: jax_svd(X_np, full_matrices=False, compute_uv=True), number=50)
		run_sec_gesvd = timeit.repeat(lambda: scipy_svd(X_np, full_matrices=False, compute_uv=True, lapack_driver="gesvd"), number=50)
		run_sec_gesdd = timeit.repeat(lambda: scipy_svd(X_np, full_matrices=False, compute_uv=True, lapack_driver="gesdd"), number=50)
		output = "({}x{}) matrix w/ {} density, Randomized: {:.3f} ms, Numpy: {:.3f} ms, JAX: {:.3f} ms, GESVD: {:.3f} ms, GESDD: {:.3f} ms".format(
			nr, nr, density,
			np.mean(run_sec_random*1000), 
			np.mean(run_sec_np*1000),
			np.mean(run_sec_jax*1000),
			np.mean(run_sec_gesvd*1000), 
			np.mean(run_sec_gesdd*1000)
		)
		print(output)
# (20x20) matrix w/ 0.01 density, Randomized: 0.104 ms, Numpy: 0.002 ms, JAX: 0.025 ms, GESVD: 0.003 ms, GESDD: 0.003 ms
# (20x20) matrix w/ 0.05 density, Randomized: 0.097 ms, Numpy: 0.003 ms, JAX: 0.022 ms, GESVD: 0.005 ms, GESDD: 0.005 ms
# (20x20) matrix w/ 0.25 density, Randomized: 0.093 ms, Numpy: 0.004 ms, JAX: 0.022 ms, GESVD: 0.009 ms, GESDD: 0.009 ms
# (20x20) matrix w/ 0.75 density, Randomized: 0.112 ms, Numpy: 0.004 ms, JAX: 0.022 ms, GESVD: 0.008 ms, GESDD: 0.008 ms
# (100x100) matrix w/ 0.01 density, Randomized: 0.149 ms, Numpy: 0.044 ms, JAX: 0.090 ms, GESVD: 0.147 ms, GESDD: 0.096 ms
# (100x100) matrix w/ 0.05 density, Randomized: 0.146 ms, Numpy: 0.070 ms, JAX: 0.093 ms, GESVD: 0.227 ms, GESDD: 0.117 ms
# (100x100) matrix w/ 0.25 density, Randomized: 0.160 ms, Numpy: 0.071 ms, JAX: 0.097 ms, GESVD: 0.231 ms, GESDD: 0.122 ms
# (100x100) matrix w/ 0.75 density, Randomized: 0.182 ms, Numpy: 0.071 ms, JAX: 0.097 ms, GESVD: 0.231 ms, GESDD: 0.119 ms
# (200x200) matrix w/ 0.01 density, Randomized: 0.158 ms, Numpy: 0.220 ms, JAX: 0.295 ms, GESVD: 1.374 ms, GESDD: 0.390 ms
# (200x200) matrix w/ 0.05 density, Randomized: 0.170 ms, Numpy: 0.261 ms, JAX: 0.331 ms, GESVD: 1.471 ms, GESDD: 0.444 ms
# (200x200) matrix w/ 0.25 density, Randomized: 0.228 ms, Numpy: 0.263 ms, JAX: 0.330 ms, GESVD: 1.525 ms, GESDD: 0.470 ms
# (200x200) matrix w/ 0.75 density, Randomized: 0.304 ms, Numpy: 0.252 ms, JAX: 0.333 ms, GESVD: 1.515 ms, GESDD: 0.454 ms

# Conclusion: Numpy LAPACK is very good all around for small, dense matrices

# %% pybind11 idea
d, D, n, J = 3, 3, int(10e3), 500
m = d*J
# U := Example dense output of A^T * < h-stacked phi_x(i) for all i in [1,n] >
U = np.random.uniform(size=(D, int(n*d)))
At = np.random.uniform(size=(D, d*J))
bn = example.BetaNuclearDense(n, d, D)

# %% Loop to get sum nuclear norm
%%time
cc = 0
nuclear_norm = 0.0
gr = np.zeros((D, d))
for _ in range(n):
	u,s,vt = np.linalg.svd(U[:,cc:(cc+d)], compute_uv=True, full_matrices=False)
	nuclear_norm += np.sum(np.abs(s))
	gr += u @ vt
	cc += d

# %% C++ version
%%time
out = bn.numpy_svd()

# %% 3x3 SVD, From paper: https://minds.wisconsin.edu/bitstream/handle/1793/60736/TR1690.pdf?sequence=1&isAllowed=y
%%time
if d == 3 and D == 3:
	out = bn.three_svd()


# %% Quality of gradient w/ landmarks / 'CRAIG'
from scipy.sparse import csc_matrix
phi_sparse_c = csc_matrix(np.hstack([phi(i) for i in range(n)]))
U = A0.T @ phi_sparse_c
G = np.zeros((D, d)) # 'True' gradient
GR = []
cc = 0
for _ in range(n):
	u,s,vt = np.linalg.svd(U[:,cc:(cc+d)], compute_uv=True, full_matrices=False)
	G += u @ vt
	GR.append(u @ vt)
	cc += d

G_dist = np.zeros(shape=(n,n))
for i in range(n):
	for j in range(n):
		G_dist[i,j] = np.linalg.norm(GR[i] - GR[j])

# %% Cont.
from tallem.landmark import landmarks

def getGreedyPerm(D):
	N = D.shape[0]
	perm = np.zeros(N, dtype=np.int64)
	lambdas = np.zeros(N)
	ds = D[0, :]
	for i in range(1, N):
		idx = np.argmax(ds)
		perm[i] = idx
		lambdas[i] = ds[idx]
		ds = np.minimum(ds, D[idx, :])
	return (perm, lambdas)
p, times = getGreedyPerm(G_dist)

gn = G / np.linalg.norm(G, axis=0)
gr = np.copy(GR[0])
grn = np.copy(GR[0])
gr_diffs = []
for i in range(1, int(n*0.50)):
	gr += GR[p[i]]
	grn = gr / np.linalg.norm(gr, axis=0)
	gr_diffs.append(np.linalg.norm(gn - grn))

# %% Plot differences in gradient 
import matplotlib.pyplot as py_plot
fig = py_plot.figure()
ax = py_plot.axes()
ax.plot(gr_diffs)
# L = landmarks(X, k = X.shape[0])['indices']