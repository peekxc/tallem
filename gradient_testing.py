# %% Quick testing 
import os
os.chdir("src/tallem")
import numpy as np
from scipy.sparse import csc_matrix
X = np.random.uniform(size=(10,2))
fast_svd.test_sparse(csc_matrix(X))

# %% Imports 
from src.tallem import TALLEM
from src.tallem.cover import IntervalCover
from src.tallem.datasets import mobius_band
from src.tallem.mds import classical_MDS
from src.tallem.distance import dist

# %% Setup parameters
X, B = mobius_band(n_polar=26, n_wide=6, embed=3).values()
B_polar = B[:,1].reshape((B.shape[0], 1))
cover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, gluing=[1])
f = lambda x: classical_MDS(dist(x, as_matrix=True), k = 2)

# %% Run TALLEM
%%time
embedding = TALLEM(cover=cover, local_map=f, n_components=3)
X_transformed = embedding.fit_transform(X, B_polar)

# %% Draw a 3D projection
import matplotlib.pyplot as plt
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], marker='o', c=B[:,0])

# %% Stiefel manifold: verify gradient 
stf = embedding._stf
pou = embedding.pou
for i in range(stf.n): 
	stf.populate_frame(i, np.sqrt(np.ravel(pou[i,:].todense())), False)

## Obtain the gradient at A0 
A0 = embedding.A0
G0 = stf.gradient(A0.T, False)[1]

## If near zero => in tangent space
(G0.T @ A0) + (A0.T @ G0)

## Test autograd version 
import autograd.numpy as auto_np
from autograd import grad
def cost_function(A):
	nuclear_norm = 0.0
	for i in range(stf.n):
		M = auto_np.array(A.T @ stf.get_frame(i)) #  A.T @ phi(j), dtype='float')
		svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False)
		nuclear_norm += auto_np.sum(auto_np.abs(svals))
	return(nuclear_norm)

## Euclidean gradient
egrad = grad(cost_function)
G0 = egrad(A0)

## If near zero => in tangent space
(G0.T @ A0) + (A0.T @ G0)

## Conclusion: neither self-computed gradient nor autograd gradient satisfies tangent vector relation 

## Try individual gradient
phi0 = stf.get_frame(0)
def cost_function(A):
	M = auto_np.array(A.T @ phi0) #  A.T @ phi(j), dtype='float')
	svals = auto_np.linalg.svd(M, full_matrices=False, compute_uv=False) 
	return(auto_np.sum(auto_np.abs(svals)))
egrad = grad(cost_function)
G0 = egrad(A0)

## If near zero => in tangent space
(G0.T @ A0) + (A0.T @ G0)

## Conclusion: individual gradient also doesn't respect tangent relation 

## Geomstats tests 
from geomstats.geometry import stiefel
J = len(embedding.cover)
St = stiefel.Stiefel(stf.d*J, stf.D)
St.belongs(A0) # true!

## NOTE! The cannonical gradient is in the tangent space! 
G_cannonical = ((G0 @ A0.T) - (A0 @ G0.T)) @ A0
(G_cannonical.T @ A0) + (A0.T @ G_cannonical) # \approx 0 all entries! 

## G_cannonical is in the tangent space at A0, and is a riemannian gradient! 
## Now turn generating a descent curve! 

## Start by generating a curve on St(n,p) via Cayley Transform
G0 = -G0
W = (G0 @ A0.T) - (A0 @ G0.T)
I = np.eye(J*stf.d)
Y = lambda tau: np.linalg.inv(I + (tau/2)*W) @ (I - (tau/2)*W) @ A0

## Verify closeness 
np.linalg.norm(Y(0)-Y(0.1))
Y(0.5).T @ Y(0.5) # should be identity! 

## Check locally the gradient changes the objective in the correct direction
stf.gradient(A0.T, False)[0]
stf.gradient(Y(1).T, False)[0]

## Search in tau in [0,1]
cost1 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 1, 100)])
cost10 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 10, 100)])
cost100 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 100, 100)])
cost1e3 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 1e3, 100)])
cost1e6 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 1e6, 100)])
cost1e9 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 1e9, 100)])
cost1e12 = np.array([stf.gradient(Y(tau).T, False)[0] for tau in np.linspace(0, 1e12, 100)])

import matplotlib.pyplot as plt
plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost1), color = "orange")
plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost10), color = "blue")
plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost100), color = "green")
plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost1e3), color = "purple")
# plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost1e6), color = "red")
# plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost1e9), color = "yellow")
plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost1e12), color = "cyan")

## Why is cost1e6 so much better!?!?! it's nearly convex! 
plt.plot(np.ravel(np.linspace(0, 1, 100)), np.ravel(cost1e3), color = "purple")

## Try to refit the embedding using manually optimized A*
from src.tallem.procrustes import global_translations
from src.tallem.assembly import assemble_frames
A_star = Y(np.linspace(0, 1e3, 100)[np.argmax(cost1e3)])
translations = global_translations(embedding.cover, embedding.alignments)
U = assemble_frames(stf, A_star, embedding.cover, embedding.pou, embedding.models, translations)


ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], marker='o', c=B[:,0])

## Optimized one is awful? 
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(U[:,0], U[:,1], U[:,2], marker='o', c=B[:,0])

## Uniform sample from the manifold in attempt to find one that is randomly better
random_costs = []
for i in range(1000):
	A_ran = St.random_uniform(1)
	random_costs.append(stf.gradient(A_ran.T, False)[0])

## Interpolate between the optimal A and the initial guess 
def close_interpolater(X, Y):
	n,p = X.shape
	I = np.eye(n)
	V = 2*Y @ np.linalg.inv(np.eye(p) + X.T @ Y)
	def interpolate(alpha: float):
		nonlocal X, V
		K = ((alpha*V) @ X.T) - (X @ (alpha*V).T) 
		return(np.linalg.inv(I - 0.5*K) @ (I + 0.5*K) @ X)
	return(interpolate)

# U = St.random_uniform(1)
Y = close_interpolater(AI, embedding.A0)

## Zero-whip span geodesic interpolator 
normalize = lambda x: x / np.linalg.norm(x) if x.ndim == 1 else x/np.sqrt(np.sum(x**2, axis = 0))

def orthonormalize(x): 
	q,r = np.linalg.qr(normalize(x))
	return(normalize(q @ np.diag(np.sign(np.diag(r)))))

def orthonormalize_by(x, y):
	if not(x.shape == y.shape): raise ValueError("Input dimensions don't match")
	x = normalize(x)
	for j in range(x.shape[1]):
		x[:,j] -= np.dot(x[:,j], y[:,j])*y[:,j]
	return(normalize(x))

Fa = np.zeros((4,2))
Fz = np.zeros((4,2))
Fa[0,0] = Fa[1,1] = 1.0
Fz[2,0] = Fz[3,1] = 1.0

def geodesic_path(Fa, Fz):
	n,p = Fa.shape
	Fa, Fz = orthonormalize(Fa), orthonormalize(Fz)
	Va, Lambda, Vzt = np.linalg.svd(Fa.T @ Fz, full_matrices = False)
	sv = Lambda[0:p][::-1]
	Va, Vz = Va[:,0:p][:,::-1], Vzt.T[:,0:p][:,::-1]
	Ga, Gz = orthonormalize(Fa @ Va), orthonormalize(Fz @ Vz)
	Gz = orthonormalize_by(Gz, Ga)
	tau = np.arccos(sv)
	bad_idx = np.where(tau < 1e-5)[0]
	if len(bad_idx) > 0:
		tau[bad_idx] = 0.0
		Gz[:,bad_idx] = Ga[:,bad_idx]
	def step_fraction(fraction):
		return(orthonormalize((Ga * np.cos(fraction*tau) + Gz * np.sin(fraction*tau)) @ Va.T))
	return(step_fraction)

Y = geodesic_path(U, A_star)

# embeddings = []
# for alpha in np.linspace(0,1,20):
# 	A_int = Y(alpha)
# 	U = assemble_frames(stf, A_int, embedding.cover, embedding.pou, embedding.models, translations)
# 	embeddings.append(U)



from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import juggle_axes
fig = plt.figure()
ax = p3.Axes3D(fig)
# Setting the axes properties
init = assemble_frames(stf, Y(0), embedding.cover, embedding.pou, embedding.models, translations)
final = assemble_frames(stf, Y(1), embedding.cover, embedding.pou, embedding.models, translations)
rng_min, rng_max = final.min(axis=0), final.max(axis=0)
ax.set_xlim3d([rng_min[0], rng_max[0]])
ax.set_ylim3d([rng_min[1], rng_max[1]])
ax.set_zlim3d([rng_min[2], rng_max[2]])

# Initial projection
points = ax.scatter(init[:,0], init[:,1], init[:,2], marker='o', c=B[:,0])
def animate(frame): 
	print(frame)
	A_int = Y(frame/100)
	U = assemble_frames(stf, A_int, embedding.cover, embedding.pou, embedding.models, translations) 
	points._offsets3d = juggle_axes(U[:,0], U[:,1], U[:,2], 'z')
anim = animation.FuncAnimation(fig, animate, frames=100, interval=50)
anim.save('basic_animation3.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()



## Frame interpolation
Fa = np.zeros((4,2))
Fz = np.zeros((4,2))
Fa[0,0] = Fa[1,1] = 1.0
Fz[2,0] = Fz[3,1] = 1.0

def givens_csr(a: float, b: float):
	if b == 0: 
		c,s,r = 1.0 if a == 0 else np.sign(a), 0.0, np.abs(a)
	elif a == 0:
		c,s,r = 0, np.sign(b), np.abs(b)
	elif np.abs(a) > np.abs(b):
		t = a / b
		u = np.sign(a) * np.sqrt(1 + t * t)
		c,s,r = 1/u, (1/u)*t, a*u
	else:
		t = a / b
		u = np.sign(b) * np.sqrt(1 + t * t)
		s,c,r = 1/u, (1/u) * t, b * u
	return(c,s,r)

def givens_rotation(i,j,c,s,n):
	G = np.eye(n)
	G[i,i], G[i,j], G[j,i], G[j,j] = c,-s,s,c
	return(G)

c,s,r = givens_csr(2.3, 1.4)
np.array([[c, -s], [s, c]]) @ np.array([2.3, 1.4]).reshape((2,1))

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8), projection='3d')
line, = ax.plot([], [], lw=2)

# call the animator.  blit=True means only re-draw the parts that have changed.


## Quasi-geodesic 
# from geomstats.geometry.stiefel import StiefelCanonicalMetric
from scipy.linalg import expm, logm 
n,p = Fa.shape
q,s,r = np.linalg.svd(Fz.T @ Fa, full_matrices = True)
R = q @ r
U_star = Fz @ R
A = logm(R.T)
q,s,vt = np.linalg.svd((np.eye(n) - Fa @ Fa.T) @ U_star, full_matrices = False)
sigma = np.arcsin(s)
quasi_geodesic = lambda t: (Fa @ vt.T @ np.diag(np.cos(t*sigma)) + q @ np.diag(t*sigma)) @ vt @ expm(t*A)


n,p = Fa.shape
q,s,rt = np.linalg.svd(Fz.T @ Fa, full_matrices = True)
R = q @ rt
a = logm(R.T)
q,s,vt = np.linalg.svd(np.real(uq @ expm(t*B) @ Iz) @ R, full_matrices = False)
sigma = np.arcsin(s)
b = np.diag(sigma) @ vt
c = -(1/6)*(b @ a @ b.T)

b11 = vt.T @ np.diag(np.cos(sigma)) @ vt @ R.T
b12 = -vt.T @ np.diag(np.sin(sigma)) @ expm(c)
b21 = np.diag(np.sin(sigma)) @ vt @ R.T
b22 = np.diag(np.cos(sigma)) @ expm(c)
B = np.vstack((np.hstack((b11, b12)), np.hstack((b21, b22))))
uq = np.hstack((Fa, q))

m = uq.shape[1] - p 
Iz = np.vstack((np.eye(p), np.zeros((m,m))))
Z = np.real(uq @ expm(t*B) @ Iz)


np.linalg.norm()

from geomstats.geometry.stiefel import StiefelCanonicalMetric
stf_cm.log(R.T)

# StiefelCanonicalMetric()
# .log(point=R.T, base_point=Fa)


## Try to interpolate using weighted broenius norm + f 
project = lambda tau: assemble_frames(stf, Y(tau), embedding.cover, embedding.pou, embedding.models, translations) 
embeddings = [project(tau) for tau in np.linspace(0, 1, 10)]


np.linalg.norm(embeddings[0])

# ## Try projecting onto St(n,p) via retraction 
# # scm = stiefel.StiefelCanonicalMetric(stf.d * J, stf.D)
# ## nope, still awful 
# ## Correction: check maximizing vs minimizing! We're maximizing! 
# q,r = np.linalg.qr(A_star)
# U2 = assemble_frames(stf, q, embedding.cover, embedding.pou, embedding.models, translations)
# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(U2[:,0], U2[:,1], U2[:,2], marker='o', c=B[:,0])

# ## verify original objective holds 
# frob_cost = 0.0
# for i in range(stf.n):
# 	u,s,vt = np.linalg.svd(q  @ q.T @ stf.get_frame(i), full_matrices = False)
# 	frob_cost += np.linalg.norm((u @ vt) - stf.get_frame(i))

# frob_cost2 = 0.0
# for i in range(stf.n):
# 	u,s,vt = np.linalg.svd(A0  @ A0.T @ stf.get_frame(i), full_matrices = False)
# 	frob_cost2 += np.linalg.norm((u @ vt) - stf.get_frame(i))

# TODO: add gradients to tox, nose, or pytest testing https://tox.readthedocs.io/en/latest/



# %% Frey faces (each of size 20 x 28)  
import scipy.io
frey = scipy.io.loadmat("/Users/mpiekenbrock/Downloads/frey_rawface.mat")
faces = frey['ff'].T

B_polar = B[:,1].reshape((B.shape[0], 1))
cover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, gluing=[1])
f = lambda x: classical_MDS(dist(x, as_matrix=True), k = 2)

# %% Run TALLEM
%%time
embedding = TALLEM(cover=cover, local_map=f, n_components=3)
X_transformed = embedding.fit_transform(X, B_polar)

x = np.random.normal(size=(100,2))
x[:,0] = x[:,0]*4.5
x[:,1] = x[:,1]*0.50
angle = np.pi/4
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
x = (R @ x.T).T
x += np.array([12, 6])

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
# plt.scatter(x[:,0], x[:,1])
plt.scatter(pca(x)[:,0], pca(x)[:,1])

# %% 
# neighborhood_graph(d, radius = 2*8.5)
neighborhood_graph(d, k = 5)


# %% 
import os
os.chdir("src/tallem")
import fast_svd
import numpy as np
X = np.random.uniform(size=(10,2))
fast_svd.test_sparse(csc_matrix(X))
