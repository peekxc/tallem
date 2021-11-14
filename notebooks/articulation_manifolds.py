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
import numpy as np
from tallem import TALLEM
from tallem.cover import LandmarkCover
from tallem.datasets import *
from tallem.samplers import landmarks

## Sample images
samples, params, blob, c = white_dot(n_pixels=17, r=0.35, n=(6500, 250), method="grid")
# ind = np.random.choice(range(samples.shape[0]), size=3*8, replace=False)
# plot_images(samples[ind,:], shape=(17,17), max_val=c, layout=(3,8), figsize=(12,4))
Lind, Lrad = landmarks(samples, k = 2500)
X = np.vstack((samples[Lind,:], samples[-250:,]))

from tallem.samplers import landmarks
cover = LandmarkCover(X, k=20, scale=1.2)
assert(np.all(np.array([len(s) for s in cover.values()]) > 1))

top = TALLEM(cover, local_map="pca3", n_components=3)
emb = top.fit_transform(X=X, B=X)


emb = pca(X, d=3)

## Eccentricity for color
# ecc = np.array([np.linalg.norm(p - np.array([0.5, 0.5, 0.0])) for p in params])
# angles = np.linspace(0, 360, num=6, endpoint=False)
# scatter3D(emb, c=ecc, angles=angles, layout=(2, 3), figsize=(18,12))

# top.plot_nerve(vertex_scale=10)

from tallem.dimred import isomap
Y = top.assemble_high()
Z = isomap(Y, 2, p = 0.15)
scatter2D(Z, figsize=(6, 6), c=Theta)


## holoviews 
import holoviews as hv
from holoviews import dim, opts
from holoviews.operation.datashader import datashade, shade, dynspread, spread, rasterize
hv.extension('matplotlib')

## 
opts.defaults(
	opts.Image(cmap="gray_r", axiswise=True),
	opts.Points(cmap="bwr", edgecolors='k', s=50, alpha=1.0), # Remove color_index=2
	opts.RGB(bgcolor="white", show_grid=False),
	opts.Scatter3D(color='z', fig_size=250, cmap='fire', edgecolor='k', s=25, alpha=0.80)
)
import colorcet as cc
from tallem.distance import dist
from tallem.color import bin_color
d = dist(np.zeros_like(X[0,:]), X)
# d = dist(emb, np.array([40.0, -40.0, -40.0]))
pt_color = list(bin_color(hist_equalize(d, number_bins=100), cc.bmy).flatten())
scatter_opts = opts.Scatter3D(azimuth=-50, elevation=25, cmap='fire', s=18, alpha = 1.00, edgecolor='none')

# q = np.quantile(d, np.linspace(0.0, 1.0, 100))
# from skimage.exposure import equalize_hist

def hist_equalize(x, scale=1000, number_bins=100000):
	h, bins = np.histogram(x.flatten(), number_bins, density=True)
	cdf = h.cumsum() # cumulative distribution function
	cdf = np.max(x) * cdf / cdf[-1] # normalize
	return(np.interp(x.flatten(), bins[:-1], cdf))

pc = hv.Table({'x': emb[:,0], 'y': emb[:,1], 'z': emb[:,2], 'd': hist_equalize(d) }, kdims=["x", "y", "z"], vdims=["d"])
hv.Scatter3D(pc).opts(scatter_opts(color='d', cmap="viridis", s=16, edgecolor='gray',linewidth=0.30))

# hv.plotting.util.list_cmaps(records=True)



from tallem.color import linear_gradient


from tallem.dimred import cmds
hv.Scatter(cmds(X, d = 2))

from tallem.dimred import isomap
iso_emb = isomap(X, d=3, p=0.15)

wut = hv.Scatter3D(iso_emb).opts(cmap=cc.bmy)

from holoviews.operation.datashader import datashade, shade, dynspread, spread, rasterize
from holoviews.operation import decimate

import colorcet as cc
# relevent color maps (from: https://colorcet.holoviz.org/user_guide/index.html)
# cc.fire := black -> dark red -> red -> orange -> yellow 
# cc.isolum := light blue -> light green -> light yellow -> light orange 
# cc.kbc := dark blue -> royal blue -> sky blue -> light blue 
# cc.bmy := dark blue -> purple -> pink -> orange -> yellow 
# cc.CET_L17 := lighter version of bmy 
# cc.CET_L18 := lighter version of fire
# cc.rainbow 
# cc.colorwheel
# cc.coolwarm := blue -> white -> red



# %% Josh's rotating strip example 
from tallem import TALLEM
from tallem.cover import *
from tallem.datasets import *
bar, c = mobius_bars(n_pixels=25, r=0.14, sigma=2.25)

## Show grid of simples
samples = []
for b in np.linspace(1.0, 0.0, num=9):
	for theta in np.linspace(0, np.pi, num=11, endpoint=False):
		samples.append(np.ravel(bar(b, theta)).flatten())
samples = np.vstack(samples)
fig, ax = plot_images(samples, shape=(25,25), max_val=c, layout=(9,11))

R = np.random.uniform(size=5000, low=0.0, high=1.0)
Theta = np.random.uniform(size=5000, low=0.0, high=np.pi)

## Generate the data 
bars = np.vstack([np.ravel(bar(b, theta)).flatten() for b, theta in zip(R, Theta)])
params = np.vstack([(b, theta) for b, theta in zip(R, Theta)])

## Landmark to get a nice sampling 
from tallem.samplers import landmarks
Lind, Lrad = landmarks(bars, 1200)
X = bars[Lind,:]
p = params[Lind,:]

## Draw random samples
idx = np.random.choice(X.shape[0], size=25*25, replace=False)
fig, ax = plot_images(X[idx,:], shape=(25,25), max_val=c, layout=(9,11))

cover = LandmarkCover(X, k=20, scale=1.1)
assert np.all(np.array([len(s) for s in cover.values()]) > 1)
assert validate_cover(X.shape[0], cover)

top = TALLEM(cover, local_map="iso3", n_components=3)
emb = top.fit_transform(X=X, B=X)

scatter3D(emb, c=p[:,1])

## Use parameters for color
# angles = np.linspace(0, 360, num=6, endpoint=False)
# scatter3D(emb, c=params[:,0], angles=angles, layout=(2, 3), figsize=(14,10))

from bokeh.io import output_notebook
output_notebook()
top.plot_nerve(vertex_scale=10, layout="hausdorff", edge_color="frame")

from tallem.dimred import isomap
Y = top.assemble_high()
Z = isomap(Y, 3, p = 0.15)
scatter3D(Z, figsize=(8, 6), c=p[:,1])

scatter3D(isomap(X, 3, p = 0.15), figsize=(8, 6), c=p[:,1])
best_iso = isomap(X, 3, p = 0.15)
scatter3D(best_iso, c=p[:,1])

## Use circular coordinate
polar_coordinate = p[:,1]
m_dist = lambda x,y: np.minimum(abs(x - y), (np.pi) - abs(x - y))
cover = IntervalCover(polar_coordinate, n_sets=12, overlap=0.60, metric=m_dist, space=[0, np.pi])

np.array([len(s) for s in cover.values()])
assert np.all(np.array([len(s) for s in cover.values()]) > 0)

top = TALLEM(cover, local_map="iso3", n_components=3, pou="triangular")
emb = top.fit_transform(X=X, B=polar_coordinate)

# angles = np.linspace(0, 360, num=6, endpoint=False)
# scatter3D(emb, c=params[:,0], angles=angles, layout=(2, 3), figsize=(14,10))
fig, ax = scatter3D(emb, c=p[:,1])
# import matplotlib.pyplot as plt
# plt.imshow(X[0,:].reshape((25,25)), aspect='auto')
# https://stackoverflow.com/questions/13570287/image-overlay-in-3d-plot-using-python
# plt.plot_surface()

from bokeh.io import output_notebook
output_notebook()
top.plot_nerve(vertex_scale=10, layout="hausdorff", edge_color="frame")


## Factorial design to try to find a good embedding 
from tallem.alignment import opa
error = {}
for k in range(10, 25):
	for lm in ["pca1", "pca2", "pca3", "iso2", "iso3"]:
		for scale in np.linspace(1.0, 1.8, 10):
			cover = LandmarkCover(X, k=k, scale=scale)
			top = TALLEM(cover, local_map=lm, n_components=3, pou="triangular")
			emb = top.fit_transform(X=X, B=polar_coordinate)
			error[(k,lm,scale)] = opa(emb, best_iso)['distance']
	print(k)
			

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
from tallem.cover import LandmarkCover
from tallem.dimred import rnn_graph
import numpy as np
x = np.random.uniform(size=(100,2))

G = rnn_graph(x, p=0.05)

def greedy_weighted_set_cover(n, S, W):
	''' 
	Computes a set of indices I \in [m] whose subsets S[I] = { S_1, S_2, ..., S_k }
	yield an approximation to the minimal weighted set cover, i.e.

		S* = argmin_{I \subseteq [m]} \sum_{i \in I} W[i]
				 such that S_1 \cup ... \cup S_k covers [n]
	
	Parameters: 
		n: int := The number of points the subsets must cover 
		S: (n x J) sparsematrix := A sparse matrix whose non-zero elements indicate the subsets (one subset per column-wise)
		W: ndarray(J,1) := weights for each subset 

	Returns: 
		C: ndarray(k,1) := set of indices of which subsets form a minimal cover 
	'''
	assert issparse(S)
	assert S.shape[0] == n and S.shape[1] == len(W)
	J = S.shape[1]

	def covered(I):
		membership = np.zeros(n, dtype=bool)
		for j in I: membership[np.flatnonzero(S[:,j].A)] = True
		return(membership)

	C = []
	membership = covered(C)
	while not(np.all(membership)):
		not_covered = np.flatnonzero(np.logical_not(membership))
		cost_effectiveness = []
		for j in range(J):
			S_j = np.flatnonzero(G[:,j].A)
			size_uncovered = len(np.intersect1d(S_j, not_covered))
			cost_effectiveness.append(size_uncovered/W[j])
		C.append(np.argmax(cost_effectiveness))
		membership = covered(C)
	
	## Return the greedy cover
	return(np.array(C))

## Estimate tangent spacees 








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
	''' Projects a set of points 'X' (row-wise) orthogonally onto the subspace spanned by the columns of U '''
	assert X.ndim == 2 and X.shape[0] == U.shape[0]
	return(U @ (U.T @ X))

# https://arxiv.org/pdf/1208.1065.pdf
# "we are interested in the mapping that orthogonally projects the manifold points in a neighborhood"


from tallem.color import linear_gradient, bin_color
col_pal = linear_gradient(["blue", "yellow", "orange", "red"], 100)['hex']
pt_col = bin_color(w, col_pal)
scatter3D(x, c=pt_col)



from numpy.typing import ArrayLike
from typing import Iterable
def tangent_spaces(X: ArrayLike, d: int, S: Iterable, S_len: int = "infer"):
	'''
		Given a set of points X in R^D representing samples in the vicinity of a d-dimensional embedded submanifold in D-dimensional 
		Euclidean space and an set of subsets S of X, this function constructs a set basis vectors spanning the optimal d-dimensional 
		linear subspaces approximating the tangent spaces for each subset s_i \in S. The vectors spanning the tangent spaces are given 
		as the first d eigenvectors of the covariance matrix of each subset of X, which are computed via PCA. See [1] for convergence 
		analysis and discussion. 

		Parameters: 
			X : ndarray(n,D) := set of points in R^D
			d : int := embedding dimension
			S : Iterable := iterable of subsets of 'X'. Each value must be a subset of { 1, 2, ..., n }

		Returns: 
			vectors : ndarray(|S|, D, d) := tangent space estimates
			error 	: ndarray(|S|) := sum of reconstruction errors for each tangent space estimate. 
		References:
			1. Tyagi, Hemant, Elıf Vural, and Pascal Frossard. "Tangent space estimation for smooth embeddings of Riemannian manifolds®." Information and Inference: A Journal of the IMA 2.1 (2013): 69-114.
	'''
	assert isinstance(X, np.ndarray) and X.ndim == 2
	normalize = lambda x: (x / np.linalg.norm(x))
	J = len(list(S)) if S_len == "infer" else S_len
	D = X.shape[1]
	vectors = np.empty(shape=(J, D, d))
	error = np.empty(J)
	for j, N in enumerate(S):
		## Get mapping from ambient dimension to embedded dimension 
		N_neig = X[N,:]
		s, U = pca(N_neig, d=d, center = True, coords=False)
		# N_neig = np.vstack([v - X[j,:] for v in N_neig])
		N_neig = N_neig - N_neig.mean(axis = 0)

		## Orthogonally project onto d-dimensional subspace 
		N_proj = np.vstack([(U @ (U.T @ np.r_['c', v])).T for v in N_neig])
		reconstruct_error = np.mean([np.sqrt(np.sum(np.power((x - x_proj), 2))) for x, x_proj in zip(N_neig, N_proj)])

		## Store vectors
		vectors[j,:,:] = U
		error[j] = reconstruct_error # s[0] #np.sum(np.abs(s))# reconstruct_error
	return(vectors, error)


#%% 'Auto'-cover using the best locally linear subsets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from tallem.dimred import rnn_graph, pca
from tallem.datasets import scatter3D
from tallem.samplers import landmarks

x, p = make_s_curve(2525, noise=0.00)
L_ind, L_rad = landmarks(x, k = 1000)
x = x[L_ind,:]
G = rnn_graph(x, p=0.20)

## Estimate tangent space
subsets = (np.flatnonzero(G[:,j].A) for j in range(G.shape[1]))
# V, w = tangent_spaces(x, 2, subsets, G.shape[1])

W = []
for i in range(G.shape[0]):
	ind = np.flatnonzero(G[:,i].A)
	Z, w = estimate_tangent(x[i,:], x[ind,:], 2)
	d = np.array([np.linalg.norm(d) for d in (x[i,:] - x[ind,:])])
	w = w/(d**2)
	W.append(np.sum(w))
W = np.array(W)

## Color by tangent space error
from tallem.color import linear_gradient, bin_color
col_pal = linear_gradient(["blue", "green", "yellow", "orange", "red"], 100)['hex']
pt_col = bin_color(W, col_pal)
fig, ax = scatter3D(x, c=pt_col)

## Pick a point. Then draw its tangent space projection 
for i in range(6):
	ind = np.flatnonzero(G[:,i].A)
	Z, w = estimate_tangent(x[i,:], x[ind,:], 2)
	fig, ax = scatter3D(x[i,:], c="black", s=50, fig=fig, ax=ax)
	fig, ax = scatter3D(Z, c="purple", fig=fig, ax=ax, s=45)
	for z, xi in zip(Z, x[ind,:]):
		line = np.vstack((z, xi)).T
		ax.plot3D(np.ravel(line[0]), np.ravel(line[1]), np.ravel(line[2]), 'gray')
	ax.text(x[i,0], x[i,1], x[i,2], size=20, zorder=1, s=f"Point: {i}")

np.sum(np.array([np.linalg.norm(d) for d in (Z - x[ind,:])])**2)

## 
T_ind = greedy_weighted_set_cover(x.shape[0], G, w)

%matplotlib
fig, ax = scatter3D(x, c=p)
scatter3D(x[T_ind,:], c="red", s=150, fig=fig, ax=ax)



# np.sum([1 for i, x in enumerate(subsets)])

fig, ax = scatter3D(x, c=p)
ax.add_collection3d(Poly3DCollection(
	verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)


# %% Geomstats 
from geomstats.geometry.hypersphere import Hypersphere
import matplotlib.pyplot as plt
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA

sphere = Hypersphere(dim=2)
data = sphere.random_von_mises_fisher(kappa=15, n_samples=140)

mean = FrechetMean(metric=sphere.metric)
mean.fit(data)

fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(data, space='S2', color='black', alpha=0.7, label='Data points')
ax = visualization.plot(mean.estimate_, space='S2', color='red', ax=ax, s=200, label='Fréchet mean')
ax.set_box_aspect([1, 1, 1])
ax.legend()

tpca = TangentPCA(metric=sphere.metric, n_components=2)
tpca = tpca.fit(data, base_point=mean.estimate_)
tangent_projected_data = tpca.transform(data)
tangent_3d = np.hstack((tangent_projected_data, np.repeat(0.0, data.shape[0])[:,None]))

from tallem.alignment import opa
pro = opa(data, tangent_3d, transform=True)

R, ss = orthogonal_procrustes(data, tangent_3d)

Q = (data @ R) + pro['translation']




R, ss = orthogonal_procrustes(tangent_3d, data)
Y = (tangent_3d - pro['translation']) @ R

## get distances
w = np.array([np.sqrt(np.sum(np.power(a-b, 2))) for a,b in zip(data, Y)])
col_pal = linear_gradient(["green", "yellow", "orange", "red"], 100)['hex']



fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(data, space='S2', color='black', alpha=0.7, label='Data points')
# ax = visualization.plot(mean.estimate_, space='S2', color='red', ax=ax, s=200, label='Fréchet mean')
# ax.scatter3D(*tangent_3d.T, c = "green")
# ax.scatter3D(*pro['coordinates'].T, c="purple")
# ax.scatter3D(*data.T, c="orange")
# ax.scatter3D(*Q.T, c="orange")
ax.scatter3D(*Y.T, c=bin_color(w, col_pal), s=30)
ax.set_box_aspect([1, 1, 1])
ax.legend()


# %% Geomstats tangent at basepoint 
import numpy as np
from scipy.linalg import orthogonal_procrustes
sphere = Hypersphere(dim=2)
data = sphere.random_von_mises_fisher(kappa=15, n_samples=140)
idx = np.argmin(np.array([np.linalg.norm(z - FrechetMean(metric=sphere.metric).fit(data).estimate_) for z in data]))
%matplotlib

from geomstats.geometry.euclidean import EuclideanMetric as em
e_metric = em(dim = 3, default_point_type="vector")

# tpca = TangentPCA(metric=sphere.metric, n_components=2)
tpca = TangentPCA(metric=e_metric, n_components=2)
tpca = tpca.fit(data, base_point=data[idx,:])
tangent_projected_data = tpca.transform(data)
tangent_3d = np.hstack((tangent_projected_data, np.repeat(0.0, data.shape[0])[:,None]))

R, ss = orthogonal_procrustes(tangent_3d, data)
Y = (tangent_3d @ R) + data[idx,:]
# w = np.array([np.linalg.norm(a - b) for a,b in zip(Y, data)])


def estimate_tangent(basepoint, points, d: int = 2, coords: bool = True):
	''' 
		Projects points onto the d-dimensional tangent plane centered at the given basepoint 
	'''
	from tallem.dimred import pca
	s, U = pca(points, d=d, center=False, coords=False)
	Z = np.vstack([(U @ (U.T @ np.r_['c', v])).T for v in (points-basepoint)])+basepoint
	w = np.array([np.linalg.norm(p-q) for p,q in zip(points, Z)])
	return(Z, w)


s, U = pca(data, d=2, center = False, coords=False)
Z = np.vstack([(U @ (U.T @ np.r_['c', v])).T for v in (data-data[idx,:])])
Z = Z + data[idx,:]

Z @ U

# R, ss = orthogonal_procrustes(Z, data)
# Z = (Z @ R)

fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(data, space='S2', color='black', alpha=0.4, s=10, label='Data points')
ax.scatter3D(*data[idx,:], c="blue", s=50, alpha=0.8)
# ax.scatter3D(*Y.T, c=bin_color(w, col_pal), alpha=0.6, s=15)
# ax.scatter3D(*tangent_3d.T, c=bin_color(w, col_pal), alpha=0.6, s=15)

# for y, x in zip(Y, data):
# 	line = np.vstack((y, x)).T
# 	ax.plot3D(np.ravel(line[0]), np.ravel(line[1]), np.ravel(line[2]), 'gray')
# ax.scatter3D(*Y.T, c="green", alpha=0.6, s=15)

for z, x in zip(Z, data):
	line = np.vstack((z, x)).T
	ax.plot3D(np.ravel(line[0]), np.ravel(line[1]), np.ravel(line[2]), 'gray')
ax.scatter3D(*Z.T, c="red", s=15, alpha=0.6)

X = Z - data[idx,:]
XX = data - data[idx,:]
ax.scatter3D(*X.T, c="orange", s=15, alpha=0.8)
ax.scatter3D(*XX.T, c="green", s=15, alpha=0.8)
for z, x in zip(X, XX):
	line = np.vstack((z, x)).T
	ax.plot3D(np.ravel(line[0]), np.ravel(line[1]), np.ravel(line[2]), 'gray')


for z, x in zip(data, X):
	line = np.vstack((z, x)).T
	ax.plot3D(np.ravel(line[0]), np.ravel(line[1]), np.ravel(line[2]), 'blue', alpha = 0.20)

# A = np.array([np.ravel((U_proj@y[:,None]).T) for y in data]) + data[idx,:]
# [(U_proj @ a[:,None]).T for a in A]
# ax.scatter3D(*A.T, c="purple", alpha=0.6, s=15)

ax.set_box_aspect([1, 1, 1])
ax.legend()


y1 = np.ravel(Y[0,:][None,:] @ U_proj)
y2 = np.ravel(y1[None,:] @ U_proj)
y3 = np.ravel(y2[None,:] @ U_proj)

z = Y[0,:] - y1
z[None,:] @ U

Z = np.array([np.ravel(y[None,:] @ U_proj) for y in data]) + data[idx,:]

from tallem.distance import dist

U_proj = U @ U.T

# R, ss = orthogonal_procrustes(Z, data)
# Z = (Z @ R)

# %% Kernel functions 
from scipy.signal import hann
from scipy.signal import convolve
phi = lambda x: np.exp(-1/(1-x**2)) # mollifier 
uni = lambda x: np.array([0.5 if abs(u) <= 1.0 else 0.0 for u in x])
epi = lambda x: (3/4)*(1 - x**2)


import matplotlib.pyplot as plt
x = np.linspace(-1.0, 1.0, 100)
plt.plot(x, phi(x), c="red")
plt.plot(x, phi(uni(x)), c="orange")
plt.plot(x, uni(x), c="blue")
plt.plot(x, epi(x), c="green")
plt.plot(x, convolve(uni(x), phi(x), mode="same")/np.sum(phi(x)), c="purple") 

## TODO: squeeze support to [-1,1]
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve
# plt.plot(x, hann(100))
# win = hann(25)
# plt.plot(x, convolve(uni(x), win, mode="same")/np.sum(win)) 

# %% Gauss curvature 
import scipy.optimize









# %% Frey faces 
import pickle
import pathlib
import scipy.io as sio
mat = sio.loadmat(pathlib.Path('~/Downloads/frey_faces.mat').expanduser().resolve())
# pickle.dump(mat['ff'], open('/Users/mpiekenbrock/tallem/data/frey_faces.pickle', 'wb'))
ff = pickle.load(open('/Users/mpiekenbrock/tallem/data/frey_faces.pickle', "rb")).T # 20 x 28

from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import *
from tallem.distance import dist 

cover = LandmarkCover(ff, k=25, scale=1.1)
top = TALLEM(cover, local_map="iso3", n_components=2)
emb = top.fit_transform(X=ff, B=ff)

s = 0.02*np.max(dist(emb))

import matplotlib.pyplot as plt
fig, ax = scatter2D(emb, alpha=0.20, s=10)


from tallem.samplers import landmarks
Lind, Lrad = landmarks(emb, k = 120)
for i in Lind: 
	bbox = np.array([emb[i,0] + s*np.array([-1.0, 1.0]), emb[i,1] + s*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(ff[i,:].reshape((28,20)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255)
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0]), right=np.max(emb[:,0]))
ax.set_ylim(bottom=np.min(emb[:,1]), top=np.max(emb[:,1]))


from tallem.dimred import pca
emb = pca(ff, 2)
Lind, Lrad = landmarks(emb, k = 120)

s = 0.02*np.max(dist(emb))
fig, ax = scatter2D(emb, alpha=0.20, s=10)
for i in Lind: 
	bbox = np.array([emb[i,0] + s*np.array([-1.0, 1.0]), emb[i,1] + s*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(ff[i,:].reshape((28,20)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255, aspect ='auto')
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0]), right=np.max(emb[:,0]))
ax.set_ylim(bottom=np.min(emb[:,1]), top=np.max(emb[:,1]))


# %% MNIST 
import pickle
import pathlib
import scipy.io as sio
# pickle.dump(mat['ff'], open('/Users/mpiekenbrock/tallem/data/frey_faces.pickle', 'wb'))
mn = pickle.load(open('/Users/mpiekenbrock/tallem/data/mnist_eights.pickle', "rb")).T # 28 x 28
rotate = lambda x: np.fliplr(x.T)
mn = np.array([rotate(mn[:,:,i]).flatten() for i in range(mn.shape[2])])
# pickle.dump(mn, open('/Users/mpiekenbrock/tallem/data/mnist_eights.pickle', 'wb'))
# mn = np.array([mn[:,:,i].flatten() for i in range(mn.shape[2])])

from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import *
from tallem.distance import dist 

cover = LandmarkCover(mn, k=15)
top = TALLEM(cover, local_map="pca2", n_components=2)
emb = top.fit_transform(X=mn, B=mn)

s = 0.02*np.max(dist(emb))

import matplotlib.pyplot as plt
fig, ax = scatter2D(emb, alpha=0.20, s=10)


from tallem.samplers import landmarks
Lind, Lrad = landmarks(emb, k = 120)
for i in Lind: 
	bbox = np.array([emb[i,0] + s*np.array([-1.0, 1.0]), emb[i,1] + s*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(mn[i,:].reshape((28,28)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255)
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0]), right=np.max(emb[:,0]))
ax.set_ylim(bottom=np.min(emb[:,1]), top=np.max(emb[:,1]))




from tallem.dimred import pca
emb = pca(mn, 2)
Lind, Lrad = landmarks(emb, k = 120)

fig, ax = scatter2D(emb, alpha=0.20, s=10)
for i in Lind: 
	bbox = np.array([emb[i,0] + s*np.array([-1.0, 1.0]), emb[i,1] + s*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(mn[i,:].reshape((28,28)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255, aspect ='auto')
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0]), right=np.max(emb[:,0]))
ax.set_ylim(bottom=np.min(emb[:,1]), top=np.max(emb[:,1]))


# plt.imshow(rotate(mn[0,:].reshape((28,28))), cmap='gray', vmin=0, vmax=255)


import numpy as np
x = np.random.uniform(size=(100,2))
y = np.random.uniform(size=(100,2))

