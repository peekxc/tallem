# %% Mobius band reference
from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import *
from tallem.datasets import *

## Get mobius band example
X, B = mobius_band()
polar_coordinate = B[:,[1]]

## Run TALLEM on interval cover using polar coordinate information
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(polar_coordinate, n_sets = 15, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)
top = TALLEM(cover, local_map="pca2", n_components=3)
emb = top.fit_transform(X, polar_coordinate)

## Rotate and view
angles = np.linspace(0, 360, num=12, endpoint=False)
scatter3D(emb, angles=angles, figsize=(16, 8), layout=(2,6), c=polar_coordinate)

top.plot_nerve(X=X, layout="hausdorff")


# %% View isomap of high-dimensional
from tallem.dimred import isomap
Y = top.assemble_high()
Z = isomap(Y, 3, p = 0.15)
scatter3D(Z, angles=angles, figsize=(16, 8), layout=(2,6), c=polar_coordinate)

#%% Rotating disk example
from tallem import TALLEM
from tallem.dimred import isomap, cmds
from tallem.cover import LandmarkCover
from tallem.datasets import rotating_disk, plot_image, plot_images, scatter2D, scatter3D

## Show exemplary disks 
disk, c = rotating_disk(n_pixels=25, r=0.15)
Theta = np.linspace(0, 2*np.pi, num=16, endpoint=False)
Disks = np.vstack([disk(theta) for theta in Theta])
plot_images(Disks, shape=(25,25), max_val=c, figsize=(18, 4), layout=(2,8))

## Generate data set
Theta = np.random.uniform(size=800, low=0.0, high=2*np.pi)
Disks = np.vstack([disk(theta) for theta in Theta])

cover = LandmarkCover(Disks, k=12, scale=1.2)
# local_map = lambda x: pca(x, d=1)
top = TALLEM(cover, local_map="pca2", n_components=2)
emb = top.fit_transform(X=Disks, B=Disks)
P = [
	cmds(Disks, d=2),
	isomap(Disks, d=2, p=0.15), 
	emb
]
scatter2D(P, layout=(1,3), figsize=(8,3), c = Theta)

# local_models = [m for m in top.models.values()]
# scatter2D(local_models, layout=(1,len(local_models)), figsize=(8,3))

top.plot_nerve(X=Disks, layout="hausdorff")

from tallem.dimred import isomap
Y = top.assemble_high()
Z = isomap(Y, 2, p = 0.15)
scatter2D(Z, figsize=(6, 6), c=Theta)

# %% White dot example
from tallem import TALLEM
from tallem.cover import LandmarkCover
from tallem.datasets import *
import numpy as np

## Sample images
samples, params, blob, c = white_dot(n_pixels=17, r=0.35, n=(600, 100), method="random")
ind = np.random.choice(range(samples.shape[0]), size=3*8, replace=False)
plot_images(samples[ind,:], shape=(17,17), max_val=c, layout=(3,8), figsize=(12,4))

from tallem.samplers import landmarks
cover = LandmarkCover(samples, k=20, scale=1.0)
assert(np.all(np.array([len(s) for s in cover.values()]) > 1))

top = TALLEM(cover, local_map="cmds3", n_components=3)
emb = top.fit_transform(X=samples, B=samples)

## Eccentricity for color
ecc = np.array([np.linalg.norm(p - np.array([0.5, 0.5, 0.0])) for p in params])
angles = np.linspace(0, 360, num=6, endpoint=False)
scatter3D(emb, c=ecc, angles=angles, layout=(2, 3), figsize=(18,12))

top.plot_nerve(vertex_scale=10)

from tallem.dimred import isomap
Y = top.assemble_high()
Z = isomap(Y, 2, p = 0.15)
scatter2D(Z, figsize=(6, 6), c=Theta)


# %% Josh's rotating strip example 
from tallem import TALLEM
from tallem.cover import *
from tallem.datasets import *
bar, c = mobius_bars(n_pixels=25, r=0.25, sigma=2.2)

## Show grid of simples
samples = []
for b in np.linspace(1.0, 0.0, num=9):
	for theta in np.linspace(0, np.pi, num=11, endpoint=False):
		samples.append(np.ravel(bar(b, theta)).flatten())
samples = np.vstack(samples)
plot_images(samples, shape=(25,25), max_val=c, layout=(9,11))

R = np.random.uniform(size=800, low=0.0, high=1.0)
Theta = np.random.uniform(size=800, low=0.0, high=np.pi)

## Dimensionality reduction
bars = np.vstack([np.ravel(bar(b, theta)).flatten() for b, theta in zip(R, Theta)])
params = np.vstack([(b, theta) for b, theta in zip(R, Theta)])

cover = LandmarkCover(bars, k=15, scale=1.0)
assert np.all(np.array([len(s) for s in cover.values()]) > 1)
assert validate_cover(bars.shape[0], cover)

top = TALLEM(cover, local_map="cmds3", n_components=3)
emb = top.fit_transform(X=bars, B=bars)

## Use parameters for color
angles = np.linspace(0, 360, num=6, endpoint=False)
scatter3D(emb, c=params[:,0], angles=angles, layout=(2, 3), figsize=(14,10))

top.plot_nerve(vertex_scale=10)

from tallem.dimred import isomap
Y = top.assemble_high()
Z = isomap(Y, 3, p = 0.15)
scatter3D(Z, figsize=(8, 6), c=params[:,0])


## Use circular coordinate
polar_coordinate = params[:,1]
m_dist = lambda x,y: np.minimum(abs(x - y), (np.pi) - abs(x - y))
cover = IntervalCover(polar_coordinate, n_sets=16, overlap=0.40, metric=m_dist, space=[0, np.pi])

assert np.all(np.array([len(s) for s in cover.values()]) > 0)

top = TALLEM(cover, local_map="cmds2", n_components=3, pou="triangular")
emb = top.fit_transform(X=bars, B=polar_coordinate)

angles = np.linspace(0, 360, num=6, endpoint=False)
scatter3D(emb, c=params[:,0], angles=angles, layout=(2, 3), figsize=(14,10))

# %% 2-sphere example 
import numpy as np
from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import LandmarkCover
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric

H = Hypersphere(2) # hs = HypersphereMetric(2)
p_ext = H.random_uniform(1500)
p_int = H.extrinsic_to_intrinsic_coords(p_ext) # both coordinates between [-1, 1]
p_int = (p_int + 1.0)/2.0
mu = p_ext.mean(axis=0)
dist_to_center = np.array([np.linalg.norm(p - mu) for p in p_ext])

## The data 
scatter3D(p_ext, c=dist_to_center)

## Tallem 
cover = LandmarkCover(p_ext, 25, scale = 1.5)
local_map = lambda x: isomap(x, d=2, k=15)
pc = TALLEM(cover, local_map=local_map, n_components=3).fit_transform(X=p_ext,B=p_ext)

P = [
	cmds(p_ext, d=3), 
	isomap(p_ext, k=15, d=3),
	pc
]

scatter3D(P, layout=(1,3), figsize=(18,6), c=dist_to_center)


# %% Klein bottle example
import scipy.io
import pickle

mat = scipy.io.loadmat('/Users/mpiekenbrock/Downloads/n50000Dct.mat')
mat['n50000Dct']

#pickle.dump(mat, open('/Users/mpiekenbrock/tallem/data/natural_images_50k_dct.pickle', 'wb'))
import pickle
import pkgutil
# pkgutil.get_data(, "data/natural_image_50k_dct.pickle")

nat_images = mat['n50000Dct']



# %% S curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve, make_swiss_roll
from tallem.datasets import scatter3D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

%matplotlib

S, p = make_s_curve(1500)
S, p = make_swiss_roll(1500)
S = S - S.mean(axis=0)

plt.ioff()
fig = plt.figure(figsize=(12, 8))
# ax = Axes3D(fig, auto_add_to_figure=False)
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
ax.scatter3D(*S.T, c=p)
ax.scatter3D(*S[0,:], c="red", s=150)
ax.add_collection3d(Poly3DCollection(R[[0,1,2],:], alpha=0.80, color="orange"))
ax.add_collection3d(Poly3DCollection(R[[0,2,3],:], alpha=0.80, color="orange"))
ax.scatter3D(*R[0,:], c="green", s=150)
ax.scatter3D(*R[1,:], c="purple", s=150)
ax.scatter3D(*R[2,:], c="blue", s=150)
ax.scatter3D(*R[3,:], c="yellow", s=150)
ax.scatter3D(*(V+S[0,:]).T, c=p, alpha=0.05)
set_axes_equal(ax)
plt.ion()
plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
%matplotlib
X = np.random.uniform(size=(100,3))
U,s,Vt = np.linalg.svd(X)
fig = plt.figure(figsize=(4, 6))
ax = plt.axes(projection='3d')
ax.scatter3D(*X.T)
points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.50], [1.0, 1.0, 0.0]])
# ax.add_collection3d(Poly3DCollection(points, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
ax.add_collection3d(Poly3DCollection([X[0], X[1], X[2], X[3], X[3]], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
plt.show()
ax.plot_surface(
	np.array([0.0, 0.0, 1.0, 1.0]), 
	np.array([0.0, 1.0, 0.0, 1.0]), 
	np.array([0.0, 0.0, 0.50, 0.0]), 
	rstride=1, cstride=1,cmap='viridis', edgecolor='none'
)

# ax = Axes3D(fig, auto_add_to_figure=False)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# %% estimate tangent space for 0th point  
from tallem.dimred import rnn_graph, pca
G = rnn_graph(S, p = 0.05)
x_neighbors = np.flatnonzero(G[0,:].A > 0)
P = S[x_neighbors,:]
evals, evecs = pca(S[x_neighbors,:], d=2, coords=False)
M = np.cov(S[x_neighbors,:], rowvar=False)
U, s, Vt = np.linalg.svd(M)


t1 = U[:,0]/np.linalg.norm(U[:,0])
t2 = U[:,1]/np.linalg.norm(U[:,1])
s = S[0,:]
ts = 10
T = np.array([s+t1*ts, s+t2*ts, s-t1*ts, s-t2*ts])
v1 = t1 + t2
v2 = -t2 + t1
v1 /= np.linalg.norm(v1)
v2 /= np.linalg.norm(v2)
R = np.array([s+v1*ts, s+v2*ts, s+(-v1)*ts,  s+(-v2)*ts])

from scipy.spatial import ConvexHull

u1, u2 = U[:,[0]], U[:,[1]]
proj_u1 = np.hstack([(np.array([s]) @ u1)*u1 for s in S]).T
proj_u2 = np.hstack([(np.array([s]) @ u2)*u2 for s in S]).T

V = proj_u1 + proj_u2
s = S[[0],:].T

# fig, ax = scatter3D(S, c=p)
ax = Axes3D(fig, auto_add_to_figure=False)


X = np.random.uniform(size=(100,3))
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(*X.T)



fig, ax = scatter3D(S, c=p)
ax = Axes3D(fig, auto_add_to_figure=False)
ax.scatter(*s, color="red", s=150)
ax.add_collection3d(Poly3DCollection(T, alpha=0.60, color="orange"))

fig, ax = scatter3D(S, angles=6, layout=(2,3), c=p, figsize=(28,16))
axes = fig.get_axes()
for ax in axes:
	ax.scatter(*s, color="red", s=150)
	
ax.scatter(*s, color="red", s=150)
ax.add_collection3d(Poly3DCollection(T, alpha=0.60, color="orange"))

fig.add_subplot(3,1,2)



# C = np.zeros((3,3))
# T = [p.reshape((len(p),1)) @ p.reshape((1,len(p))) for p in P]
# for t in T: C+= t



# Helix equation
t = np.linspace(0, 10, 50)
x, y, z = np.cos(t), np.sin(t), t

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers')])
fig.show()


# %% Weighted Set Cover solution
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from tallem.dimred import rnn_graph, pca

S, p = make_s_curve(250, noise=0.15)
G = rnn_graph(S, p = 0.05)
d = 2 ## dimension of local tangent space! 

subset_weights = []
for i, N_i in enumerate(G):
	x_neighbors = np.append(np.flatnonzero(N_i.A), i)
	if len(x_neighbors) > 1:
		M = np.cov(S[x_neighbors,:], rowvar=False)
		U, s, Vt = np.linalg.svd(M, full_matrices=False)
		
		## Use largest *d* subspaces as representation of tangent space
		U = U[:,:d]
		s_i = S[x_neighbors,:]
		
		## Project onto each subspace using inner product as weights
		proj_u = np.array([np.sum(U*v, axis = 1) for v in (s_i @ U)])
		
		## Absolute reconstruction error <=> euclidean distance from point to projection onto tangent plane 
		weight = np.sum(np.sqrt(np.sum(abs(proj_u - s_i)**2, axis = 1)))
	else: 
		weight = 0.0
	subset_weights.append(weight)

tangent_error = np.array(subset_weights)


## TODO: try weighted Greedy solution
# http://pages.cs.wisc.edu/~shuchi/courses/880-S07/scribe-notes/lecture03.pdf


## TODO: try out randomized LP solution 
# Let U = { u_1, u_2, ..., u_n } denote the universe/set of elements to cover
# Let S = { s_1, s_2, ..., s_m } denote the set of subsets with weights w = { w_1, w_2, ..., w_m }
# Let y = { y_1, y_2, ..., y_m } denote the binary variables s.t. y_i = 1 <=> s_i included in solution 
# The goal is to find the assignment Y producing a cover S* of minimal weight 
# If k > 0 is specified and one wants |S*| = k, then this is the decision variant of the weighted set cover problem
# The ILP formulation: 
# 	minimize sum_{s_j in S} w_j*s_j
# 	subject to
# 		sum_{x in S, s_j in S} y_j >= 1 for each x \in X
# 		y_j \in { 0, 1 } 
# 
# The ILP can be relaxed to LP via letting y_j vary \in [0, 1].
# Let A \in R^{n x m} where A(i,j) = 1 if u_i \in s_j and b = { 1, 1, ..., 1 } w/ |b| = m. 
# The corresponding LP is: 
# 	minimize 			 w^T * y
# 	subject to 		 Ay >= b 
# 								 0 <= y_j <= 1 		for all j in [m]
#
# From here, we have two strategies (http://theory.stanford.edu/~trevisan/cs261/lecture08.pdf): 
#   1. Rounding approach: If we know each u belongs to at most *k* sets, then we may choose
#  		 y_j = 1 if y_j >= 1/k and y_j = 0 otherwise. This guarentees a feasible cover which is 
# 		 a k-approximation of the objective y*.  
#   2. Randomized approach: Interpret each y_j as probability of including subset s_j, sample from Y*
#			 using y_j as probabilities. Due to LP, expected cost of resulting assignment is upper bounded 
#      the weight of cost(Y*). 
# Note that (1) can only be used if one knows each u belongs to at most *k* sets. However, even in this case 
# if this is known, k may be large, yielding a poor approximation (k = n/2 => ) (see https://www.cs.cmu.edu/~anupamg/adv-approx/lecture2.pdf). 
# Suppose we solve the LP once, obtaining y. Interpreting each y \in [0,1], we let C = {} and repeat:
#   - While C does not cover U:
# 	- 	For each j in [m]:
# 	- 		Assign y_j = 1 w/ probability y_j
# Assume n >= 2. Then:
# 	a. P(u_i not covered) after k iterations = e^{-k} => take e.g. k = c lg n, then P(u_i not covered) <= 1/(n^c)
# 	b. P(there exists u_i not covered) <= \sum\limits_{i=0}^n P(u_i no covered) = n*(1/n^c) = n^(1-c). 
# Thus, for k = c lg(n). One can show that the randomized approach: 
# 	=> produces a feasible cover after k iterations with probability >= 1/(1 - n^(1-c))
# 	=> produces an assignment \hat{y}* with expected cost c*ln(n)*opt(y*)
# In particular, if c = 2 then then with probability p = 1 - 1/n, we will have a 2*ln(n)-approximation after 2*lg(n) iterations.
# At best, we have a \Theta(ln n)-approximation using the LP rounding. 
# (Ideally we wanted (1-\eps)lg(n) approximation! )


def proj_subspace(X: ArrayLike, U: ArrayLike):
	''' Projects a vector 'v' orthogonally onto the subspace spanned by the columns of U '''
	assert X.ndim == 2 and X.shape[0] == U.shape[0]
	return(U @ (U.T @ X))



# %% 
import mayavi
from mayavi.mlab import points3d


points3d(X)
