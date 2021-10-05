# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

# %% 
from src.tallem.samplers import landmarks
from src.tallem.distance import dist

db = dist(B, metric=m_dist)
L, R = landmarks(db, k=10)
cover = LandmarkCover(db, 15)

D = dist(B, as_matrix=True, metric=m_dist)
cover = LandmarkCover(D, 15)

[len(cover[k]) for k in cover.keys()]

m_dist(X[0,:], X[1,:])

# %% Mobius band example again 
import numpy as np
from src.tallem.datasets import mobius_band
from src.tallem.cover import IntervalCover, LandmarkCover
from src.tallem import TALLEM

M = mobius_band(plot=False, embed=6)
X, B = M['points'], M['parameters'][:,[1]]

## Run TALLEM on polar coordinate cover 
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(B, n_sets = 10, overlap = 0.20, space = [0, 2*np.pi], metric = m_dist)
embedding = TALLEM(cover, local_map="pca2", n_components=3).fit_transform(X, B)

## Plot 
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*embedding.T, c=B)

# %% Landmark cover 
## Run TALLEM on landmark cover
from src.tallem.distance import dist
from scipy.sparse import csc_matrix
cover = LandmarkCover(dist(B, metric=m_dist), k=15, scale=1.5)

## Need to compute POU ourselves
Q = cover._neighbors.tocoo()
Q.data = 1.0 - (Q.data/cover.cover_radius) ## apply triangular pou
Q = Q / np.sum(Q, axis = 1)
P = csc_matrix(Q)

## Note: must pass in a PoU! 
embedding = TALLEM(cover, local_map="cmds2", n_components=3, pou=P).fit_transform(X, B)

## Plot 
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(52.0, 75.5)
ax.scatter3D(*embedding.T, c=B)

# %% Multivariate normal code + code to generate a pixel
### TODO: simplify these with just numpy functions to use autograd with them!
def mvn_density(locations, mu, Sigma):
	d, k, S_inv, mu = np.linalg.det(Sigma), len(mu), np.linalg.inv(Sigma), np.asanyarray(mu)[:,np.newaxis]
	result = np.zeros(locations.shape[0])
	for i, x in enumerate(locations): 
		x = x.reshape(mu.shape)
		f_x = np.exp(-0.5*((x - mu).T @ S_inv @ (x - mu)))/np.sqrt((2*np.pi)**k * d)
		result[i] = f_x
	return(result)
	
def gaussian_pixel(center, Sigma, d=17,s=1):
	from src.tallem.utility import cartesian_product
	grid_ind = cartesian_product([range(d), range(d)])
	grid = np.zeros(shape=(d,d))	
	F = mvn_density(grid_ind, center, Sigma)
	for i, ind in enumerate(grid_ind):
		r,c = ind
		grid[r,c] = s*F[i]
	return(grid)

# %% Autograd'd pixel
import autograd.numpy as auto_np
from autograd import jacobian
scale, image_sz, s = 1, (17, 17), 1
denom = np.sqrt((2*np.pi)**2)
def gaussian_pixel2(center):
	x, y = auto_np.meshgrid(auto_np.arange(image_sz[0]), auto_np.arange(image_sz[0]))
	grid = auto_np.exp(-0.5*((x - center[0])**2 + (y - center[1])**2))/denom
	return(auto_np.ravel(grid).flatten())

# jacobian(gaussian_pixel2)(auto_np.array([0.,0.]))
# from torch.autograd.functional import jacobian as t_jacobian

# denom = np.sqrt((2*np.pi)**2)
# def gaussian_pixel2_torch(center):
# 	x, y = torch.autograd.torch.meshgrid(torch.autograd.torch.arange(image_sz[0]), torch.autograd.torch.arange(image_sz[0]))
# 	grid = torch.autograd.torch.exp(-0.5*((x - center[0]) + (y - center[1])))/denom
# 	return(torch.autograd.torch.ravel(grid))

# center = torch.tensor([0.0,0.0])
# inputs = torch.rand(2, 2)
# J = t_jacobian(gaussian_pixel2_torch, inputs=center, strict=True)



# %% Generate data
import numpy as np
import random
import scipy.stats
var = 1
sigma = np.diag([var,var])
odd = lambda x: int(x if (x % 2) == 1 else (x + 1))
blob_diam = np.ceil(abs(scipy.stats.norm.ppf(0.01, scale=var) - scipy.stats.norm.ppf(0.99, scale=var)))
n, m = 1900, 100
inner_sz = odd(2*blob_diam)
extra = np.ceil(blob_diam/2)
outer_sz = odd(inner_sz+2*extra) 
center = [np.ceil(outer_sz/2)-1,np.ceil(outer_sz/2)-1]

w = (center[0] - abs(center[0] - np.array(list(range(0, outer_sz)))))**2
w = abs(np.cos(np.linspace(0, 4*np.pi, outer_sz)))
X = random.choices( population=range(0, outer_sz), weights=w, k=n )
Y = random.choices( population=range(0, outer_sz), weights=w, k=n )
# X = np.array(np.random.uniform(low = 0, high = outer_sz, size = n), dtype=int)
# Y = np.array(np.random.uniform(low = 0, high = outer_sz, size = n), dtype=int)
sphere_images = [gaussian_pixel([x,y], sigma, d=outer_sz, s=1) for x,y in zip(X, Y)]
center_images = [gaussian_pixel(center, sigma, d=outer_sz, s=r) for r in np.linspace(0, 1, 100)]

## Include only inner-grids of images
ind = range(int(extra), int(outer_sz - extra))
spheres_inner = [S[np.ix_(ind, ind)] for S in sphere_images]
centers_inner = [S[np.ix_(ind, ind)] for S in center_images]
N = np.vstack((
	np.asanyarray([np.ravel(p) for p in spheres_inner]), 
	np.asanyarray([np.ravel(p) for p in centers_inner])
))

# %% Double-check: should be center pixel
P = gaussian_pixel(center, sigma, d=outer_sz, s=1)[np.ix_(ind, ind)]
plt.imshow(P, cmap='gray', vmin=0, vmax=max_val)
fig.gca().axes.get_xaxis().set_visible(False)
fig.gca().axes.get_yaxis().set_visible(False)

plt.imshow(centers_inner[-1], cmap='gray', vmin=0, vmax=max_val)
fig.gca().axes.get_xaxis().set_visible(False)
fig.gca().axes.get_yaxis().set_visible(False)

plt.imshow(sphere_images[326], cmap='gray', vmin=0, vmax=max_val)
fig.gca().axes.get_xaxis().set_visible(False)
fig.gca().axes.get_yaxis().set_visible(False)

sphere = gaussian_pixel(center, sigma, d=outer_sz, s=1)
plt.imshow(sphere[np.ix_(ind, ind)], cmap='gray', vmin=0, vmax=max_val)
fig.gca().axes.get_xaxis().set_visible(False)
fig.gca().axes.get_yaxis().set_visible(False)

# %% Visualize the data set
max_val = mvn_density(np.array([[0,0]]), [0, 0], sigma)[0]
fig = plt.figure(figsize=(8, 8))
sample_ind = np.array(np.floor(np.random.random_sample(30)*n), dtype=int)
for i, p in enumerate(sample_ind):
	fig.add_subplot(5, 10, i+1)
	plt.imshow(spheres_inner[p], cmap='gray', vmin=0, vmax=max_val)
	fig.gca().axes.get_xaxis().set_visible(False)
	fig.gca().axes.get_yaxis().set_visible(False)


## For some reason, no images are centered
# plt.hist(np.array([np.linalg.norm(S - centers_inner[-1]) for S in spheres_inner]))

# %% PH 

from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(np.asanyarray([np.ravel(p) for p in spheres_inner]))['dgms']
plot_diagrams(diagrams, show=True)

# %% Measure KNN distance to color points by
from src.tallem.distance import dist
D = dist(N, as_matrix=True)
knn_dist = np.apply_along_axis(lambda x: np.sort(x)[15], 1, D)

# %% TALLEM
from src.tallem import TALLEM
from src.tallem.cover import LandmarkCover
from src.tallem.dimred import isomap
cover = LandmarkCover(N, 15).construct(N)

# local_map = lambda x: isomap(x, k = 15)
pc = TALLEM(cover, local_map="pca3", n_components=3).fit_transform(N)

# pc = TALLEM(cover, local_map="pca3", n_components=3)._profile(N)
# pc = TALLEM(cover, local_map="mmds3", n_components=3).fit_transform(N)
# pc = TALLEM(cover, local_map="nmds3", n_components=3).fit_transform(N)
# pc = TALLEM(cover, local_map="cmds3", n_components=3).fit_transform(N)

# xy = mmds(N, 2)
# plt.scatter(*xy.T)

# %% profile 
TALLEM(cover, local_map="pca3", n_components=3)._profile(X=N)




# %% Jacobian rejection sampling idea
import autograd.numpy as auto_np
from autograd import grad, jacobian
import numpy as np

## Test disk parameterization
f = lambda x : (x[0], x[1]*auto_np.sqrt(1.0-x[0]**2))
jf = lambda s, t: (np.sqrt(1-s**2))

from src.tallem.utility import cartesian_product
JF = [jf(s,t) for s,t in cartesian_product((np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)))]
	

def f(x): return(auto_np.array([x[0], x[1]*auto_np.sqrt(1.-x[0]**2)]))
x = auto_np.array([0., 0.])
J = jacobian(f)
np.linalg.det(J(x))

# X = np.linspace(-1, 1, 100)
# jf = lambda J: np.linalg.det(J)
# D = np.array([jf(J(auto_np.array([s, t]))) for s,t in cartesian_product((X, X))])
# ind = np.array(np.nonzero([math.isnan(d) for d in D])[0], dtype=int)
# D[ind] = 0.0


from typing import * 
from numpy.typing import ArrayLike
def rejection_sampler(
	parameterization: Callable[ArrayLike, ArrayLike],
	jacobian: Callable[ArrayLike, ArrayLike],
  min_params: ArrayLike,
  max_params: ArrayLike,
  max_jacobian: float):
	'''
	Makes a rejection sampler of a parameterized embedding from a parameter space to a coordinate space which 
	using the jacobian determinant of the map to make samples from the output coordinate space approximately 
	uniform with respect to a change-of-coordinate Lebesgue measure...

	Parameters: 
		1. parameterization - a function which accepts an (n x d) matrix of intrinsic parameters and returns an (n x p) matrix of extrinsic parameters
		2. jacobian - Jacobian determinant function. Must take as input parameters from (1) and output a set floats for each set of parameters 
		3. min_params - lower bounds on the each of the intrinsic parameters
		4. max_params - upper bounds on the each of the intrinsic parameters
		5. max_jacobian - an upper bound on the Jacobian determinant. The tightness of this bound affects the quality of the sampling.

	This function was inspired by the 'make_rejection_sampler' function from the R package 'tdaunif' by Cory Brunson. 
	'''
	d, p = len(min_params), len(parameterization(min_params)) # read-only, no need for nonlocal
	def sampler(n: int) -> ArrayLike:
		output_samples = np.zeros(shape=(0, d))
		while (output_samples.shape[0] < n):
			sample_params = np.c_[[np.random.uniform(low=min_params[j], high=max_params[j], size=n) for j in range(d)]].T
			j_vals = np.array([jacobian(params) for params in sample_params])
			density_threshold = np.random.uniform(low=0,high=max_jacobian,size=n)
			output_samples = np.append(output_samples, sample_params[j_vals > density_threshold,:], axis=0)
		#return(output_samples[:n, :])
		return(np.asanyarray([parameterization(sample) for sample in output_samples[:n, :]]))
	return(sampler)

## Test disk sampler 
def f(x): return(auto_np.array([x[0], x[1]*auto_np.sqrt(1.-x[0]**2)]))# parameterization

import math
def J_determinant(f): 
	J = jacobian(f)
	def det(x: ArrayLike): 
		jac = J(x)
		if jac.shape[0] == jac.shape[1]:
			D = np.linalg.det(jac)
		else: 
			if jac.shape[0] < jac.shape[1]:
				D = np.sqrt(np.linalg.det(jac @ jac.T))
			else: 
				D = np.sqrt(np.linalg.det(jac.T @ jac)) 
		return(D if not(math.isnan(D)) else 0.0)
	return(det)

## autograd determinant
J_det = J_determinant(f)

## closed-form 
J_det = lambda x: np.linalg.det(np.array([[1, 0],[0, np.sqrt(1.0-x[0]**2)]]))

# J_det(auto_np.array([0.0, 0.0]))

sampler = rejection_sampler(
	parameterization = f, 
	jacobian = J_det, 
	min_params = [-1.0, -1.0],
	max_params = [1.0, 1.0],
	max_jacobian = 1.0
)

#%% Plot it! 
D2 = sampler(1000)
# D2 = np.asanyarray([f(d) for d in D2])
import matplotlib.pyplot as plt
plt.scatter(*D2.T)

import seaborn
import pandas
seaborn.pairplot(pandas.DataFrame(D2))

# %% Torus example
import seaborn
import pandas

r, R = 0.9, 1.0
def f(x):
	theta, phi = x	
	e1 = (R + r*auto_np.cos(theta))*auto_np.cos(phi)
	e2 = (R + r*auto_np.cos(theta))*auto_np.sin(phi)
	e3 = r*auto_np.sin(theta)
	return(auto_np.array([e1, e2, e3]))

theta = np.random.uniform(low=0, high=2*np.pi, size=2500)
phi = np.random.uniform(low=0, high=2*np.pi, size=2500)
T = np.asanyarray([f(x) for x in np.c_[phi, theta]])
seaborn.pairplot(pandas.DataFrame(T), height=7)

## Uniformly sampled torus using autograd 
# J_det = J_determinant(f)
# J_det = lambda x: (r**2) * (R + r*np.cos(x[0]))**2
# J_det = lambda x: (1 + (r/R)*np.cos(x[0]))/(2*np.pi)

## Double-check
# J_det(auto_np.array([0., 0.]))
# J_det(auto_np.array([np.pi, np.pi]))

sampler = rejection_sampler(
	parameterization = f, 
	jacobian = J_det, 
	min_params = [0, 0],
	max_params = [2*np.pi, 2*np.pi],
	max_jacobian = (r**2)*(R + r)**2
)
# det() = r**2(R + r cos(θ))**2
T2 = sampler(2500)
seaborn.pairplot(pandas.DataFrame(T2), height=7)

# %% Try stratified sampling on torus
bins = 8
# strata <- hist(x = f, breaks = seq(min(f), max(f), length.out = n.strata+1L), plot = FALSE)
# ns <- table(sample(x = seq(n.strata), size = m, replace = TRUE, prob = strata$counts/length(f)))

## Suppose we want 1000 samples. We calculate the area of each cell/stratum in the strafication of 
## [0,1]^2, then obtain how many samples should exist in each strata. We then make a rejection sampler
## constructed *on the bounds of the strata of interest*, sampling how ever many points needed according 
## to the original input parameterization. The rejection sampler *should* fix the warping within that strata. 
n = 2500
bins = 4
B = bins**2 
n_per_stratum = int(np.floor(n / B))
n_extra = n % B

from src.tallem.utility import cartesian_product

cell_min = np.linspace(0, 2*np.pi, 4, endpoint=False)
cell_max = np.linspace(cell_min[1], 2*np.pi, 4, endpoint=True)

samples = []
for i in range(bins):
	for j in range(bins):
		sampler = rejection_sampler(
			parameterization = f, 
			jacobian = J_det, 
			min_params = [cell_min[i], cell_min[j]],
			max_params = [cell_max[i], cell_max[j]],
			max_jacobian = (r**2)*(R + r)**2
		)
		samples.append(sampler(n_per_stratum))

samples = np.vstack(samples)
seaborn.pairplot(pandas.DataFrame(samples), height=7)

# cell_area = (1.0/bins)**2
# cell_area*1000



## The idea 
# j_vals <- do.call(jacobian, args = param_vals)
# density_threshold <- runif(n, 0, max_jacobian)
# x <- rbind(x, param_vals[j_vals > density_threshold, , drop = FALSE])

# %% Visualize tallem result
import matplotlib.pyplot as plt 
from src.tallem.color import linear_gradient, bin_color
col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

for angle in range(0, 360, 15):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*pc.T, color=bin_color(knn_dist, col_pal))
	ax.view_init(30, angle)
	plt.pause(2.50)


#%% Uniformly distributed samples 
import numpy as np
import random
import scipy.stats
var = 1
sigma = np.diag([var,var])
odd = lambda x: int(x if (x % 2) == 1 else (x + 1))
blob_diam = np.ceil(abs(scipy.stats.norm.ppf(0.01, scale=var) - scipy.stats.norm.ppf(0.99, scale=var)))
n, m = 1900, 100
inner_sz = odd(2*blob_diam)
extra = np.ceil(blob_diam/2)
outer_sz = odd(inner_sz+2*extra) 
center = [np.ceil(outer_sz/2)-1,np.ceil(outer_sz/2)-1]

w = (center[0] - abs(center[0] - np.array(list(range(0, outer_sz)))))**2
w = abs(np.cos(np.linspace(0, 4*np.pi, outer_sz)))
X = random.choices( population=range(0, outer_sz), weights=w, k=n )
Y = random.choices( population=range(0, outer_sz), weights=w, k=n )
# X = np.array(np.random.uniform(low = 0, high = outer_sz, size = n), dtype=int)
# Y = np.array(np.random.uniform(low = 0, high = outer_sz, size = n), dtype=int)
sphere_images = [gaussian_pixel([x,y], sigma, d=outer_sz, s=1) for x,y in zip(X, Y)]
center_images = [gaussian_pixel(center, sigma, d=outer_sz, s=r) for r in np.linspace(0, 1, 100)]

# jacobian(gaussian_pixel2)

J_det = J_determinant(gaussian_pixel2)
J_det(auto_np.array([14.0,14.1]))

from src.tallem.utility import cartesian_product
jd = np.array([J_det(np.array(p, dtype=float)) for p in cartesian_product((range(image_sz[0]), range(image_sz[1])))])

sampler = rejection_sampler(
	parameterization = gaussian_pixel2, 
	jacobian = J_det, 
	min_params = [0.0, 0.0],
	max_params = image_sz,
	max_jacobian = np.max(jd)
)

images = sampler(15)
images[0,:].reshape(image_sz)

for i in range(1, 15):
	fig.add_subplot(3, 5, i)
	plot_image(images[i,:].reshape(image_sz))

def plot_image(P, max_val = "default"):
	if max_val == "default": max_val = np.max(P)
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(8, 8))
	plt.imshow(P, cmap='gray', vmin=0, vmax=max_val)
	fig.gca().axes.get_xaxis().set_visible(False)
	fig.gca().axes.get_yaxis().set_visible(False)


images = sampler(1000)

#%% 
from src.tallem.cover import LandmarkCover
from src.tallem import TALLEM

c_images = np.vstack([np.ravel(img).flatten() for img in center_images])
images = np.vstack((images, c_images))

cover = LandmarkCover(images, k=15).construct(images)

# %% 
%%time
embedding = TALLEM(cover, local_map="pca3", n_components=3).fit_transform(images, images)

# %% 
from src.tallem.distance import dist
D = dist(embedding, as_matrix=True)
knn_dist = np.apply_along_axis(lambda x: np.sort(x)[15], 1, D)

from src.tallem.color import bin_color, linear_gradient
col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

 # color=bin_color(knn_dist, col_pal)
for angle in np.linspace(0, 360, 10):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*embedding.T, color=bin_color(knn_dist, col_pal))
	ax.view_init(30, angle)
	plt.pause(0.50)

# len(reduce(np.union1d, list(cover.values())))


## WRONG
np.linalg.norm(gaussian_pixel2(auto_np.array([0.1,0.1])) - gaussian_pixel2(auto_np.array([0.1,0.1])))







# %% Mobius band example again 
import numpy as np
from src.tallem.datasets import mobius_band
from src.tallem.cover import IntervalCover
from src.tallem import TALLEM

M = mobius_band(plot=False, embed=6)
X, B = M['points'], M['parameters'][:,[1]]

## Run TALLEM on polar coordinate cover 
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(B, n_sets = 10, overlap = 0.20, space = [0, 2*np.pi], metric = m_dist)
embedding = TALLEM(cover, local_map="pca2", n_components=3).fit_transform(X, B)

# %% Debug partition of unity
cover = IntervalCover(B, n_sets = 10, overlap = 0.20, space = [0, 2*np.pi], metric = m_dist) 
cover.sets
# cover.construct(B, (0))
# cover.set_distance(B, (0))
# np.nonzero(cover.set_distance(B, (0)) <= 1.0)


# %% Profiling TALLEM 
import line_profiler
import numpy as np
profile = line_profiler.LineProfiler()
def do_func():
	c = 0.0
	for i in list(range(10000)):
		c += np.sin(np.sqrt((i + 1)*5))
	return(c)
profile.add_function(do_func)
profile.enable_by_count()
do_func()
profile.print_stats(output_unit=1e-3)


# %% Geomstats uniform sphere sampling 
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
H = Hypersphere(2) # hs = HypersphereMetric(2)
p_ext = H.random_uniform(1900)
p_int = H.extrinsic_to_intrinsic_coords(p_ext) # both coordinates between [-1, 1]
p_int = (p_int + 1.0)/2.0


var = 1
sigma = np.diag([var,var])
odd = lambda x: int(x if (x % 2) == 1 else (x + 1))
blob_diam = np.ceil(abs(scipy.stats.norm.ppf(0.01, scale=var) - scipy.stats.norm.ppf(0.99, scale=var)))
n, m = 1900, 100
inner_sz = odd(2*blob_diam)
extra = np.ceil(blob_diam/2)
outer_sz = odd(inner_sz+2*extra) 
center = [np.ceil(outer_sz/2)-1,np.ceil(outer_sz/2)-1]

sphere_images = [gaussian_pixel([x,y], sigma, d=outer_sz, s=1) for x,y in (p_int * outer_sz)]
center_images = [gaussian_pixel(center, sigma, d=outer_sz, s=r) for r in np.linspace(0, 1, 100)]

## Include only inner-grids of images
ind = range(int(extra), int(outer_sz - extra))
spheres_inner = [S[np.ix_(ind, ind)] for S in sphere_images]
centers_inner = [S[np.ix_(ind, ind)] for S in center_images]
N = np.vstack((
	np.asanyarray([np.ravel(p) for p in spheres_inner]), 
	np.asanyarray([np.ravel(p) for p in centers_inner])
))


from src.tallem import TALLEM
from src.tallem.cover import LandmarkCover
from src.tallem.dimred import isomap

cover = LandmarkCover(N, 15).construct(N)
pc = TALLEM(cover, local_map="pca3", n_components=3).fit_transform(N)

from src.tallem.distance import dist
D = dist(N, as_matrix=True)
knn_dist = np.apply_along_axis(lambda x: np.sort(x)[15], 1, D)

import matplotlib.pyplot as plt 
from src.tallem.color import linear_gradient, bin_color
col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

for angle in range(0, 360, 30):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*pc.T, color=bin_color(knn_dist, col_pal))
	ax.view_init(30, angle)
	plt.pause(1.50)


# %% optimizing the assembly 
import numpy as np
from src.tallem.datasets import mobius_band
from src.tallem.cover import IntervalCover, LandmarkCover
from src.tallem import TALLEM

M = mobius_band(n_polar=66, n_wide=9, scale_band = 0.25, plot=True, embed=3)
X, B = M['points'], M['parameters'][:,[1]]

## Run TALLEM on polar coordinate cover 
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(B, n_sets = 10, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)
# cover = LandmarkCover(B, k = 10, metric = m_dist).construct(B)

top = TALLEM(cover, local_map="pca2", n_components=3)
top.fit(X, B)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*top.embedding_.T, c = B)

# id0 = np.nonzero(cover.set_distance(B, (0,)) <= 1.0)
# id1 = np.nonzero(cover.set_distance(B, (1,)) <= 1.0)
# int_pts, ind0, ind1 = np.intersect1d(id0, id1, return_indices=True)




# models = { index : top.local_map(X[np.array(subset),:]) for index, subset in cover.items() }
# from src.tallem.cover import partition_of_unity
# pou = partition_of_unity(B, cover = cover, similarity = "triangular") 
# pou[1,:].todense()




# %% Uniform sampling white dot 
from src.tallem.datasets import gaussian_pixel2
blob = gaussian_pixel2(d=0.25, n_pixels=17)

plot_image(blob([0.5, 0.5]))
plot_image(blob([0, 0]))
plot_image(blob([0, 1.0]))
plot_image(blob([1.0, 0.0]))

# %% MVN contours 
M = np.meshgrid(range(100), range(100))
ind = np.ix_(range(10), range(10))

def mvn_density(x, mu, Sigma):
	x, mu = np.asanyarray(x), np.asanyarray(mu)
	if x.ndim == 1: np.array(x).reshape((len(x), 1))
	if mu.ndim == 1: np.array(mu).reshape((len(mu), 1))
	Sigma_inv = np.linalg.inv(Sigma)
	denom = np.sqrt(((2*np.pi)**2) * np.linalg.det(Sigma))
	return(np.exp((-0.5*((x - mu).T @ Sigma_inv @ (x - mu))))/denom)
	# np.exp(-0.5*(sigma_inv * ((x-mu[0])**2 + (y-mu[1])**2)))/denom

## Contour on [0,5] x [0, 5] w/ blob radius = 1.0 => sigma = 1/3
## blob diameter d := 6*sigma, r := 3*sigma 
c = 3.090232
Sigma = np.diag([1/c, 1/c])
mu = np.array([2.5, 2.5])

Z = np.zeros(shape=(100,100))
X, Y = np.linspace(0, 5, 100), np.linspace(0, 5, 100)
for i, x in enumerate(X):
	for j, y in enumerate(Y):
		Z[i,j] = mvn_density([x, y], mu, Sigma)

## Looks good 
import matplotlib.pyplot as plt
X,Y = np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100))
plt.contourf(X, Y, Z, levels=np.linspace(0, 1, 20))

## Make blob version on the domain [0,1] x [0,1]
import autograd.numpy as auto_np
def gaussian_blob(n_pixels, r):
	sd = r/3.090232
	sigma = sd**2
	sigma_inv = 1.0/sigma
	denom = np.sqrt(((2*auto_np.pi)**2) * (sigma**2))
	def blob(mu): # mu can be anywhere; center of image is [0.5, 0.5]
		loc = auto_np.linspace(0, 1, n_pixels, False) + 1/(2*n_pixels)
		x,y = auto_np.meshgrid(loc, loc)
		grid = auto_np.exp(-0.5*(sigma_inv * ((x-mu[0])**2 + (y-mu[1])**2)))/denom
		return(auto_np.ravel(grid).flatten())
	return(blob, auto_np.exp(0)/denom)
# Differentiable version

p = 0.25 # blob radius as proportion of image (between 0 < p < 1)
blob, nc = gaussian_blob(17, 0.25)
# plot_image(blob([0.5, 0.5]).reshape((17,17)), nc)
# plot_image(blob([-0.25, 0]).reshape((17,17)), nc)
# plot_image(blob([-0.25, 0]).reshape((17,17)), nc)
# plot_image(blob([-0.05, -0.05]).reshape((17,17)), nc)

x = np.random.uniform(low=-p, high=1+p, size=(1000,2))

## Generate uniform samples including blob outside image
blob_images = np.vstack([blob(mu) for mu in x])

# Test case -- use jacobian determinant below
# from autograd import jacobian
# J_blob = jacobian(blob)
# J_blob(auto_np.array([0.5, 0.5]))# works !

J_det = J_determinant(blob)
max_det = np.max(np.array([J_det(auto_np.array(mu)) for mu in x]))

sampler = rejection_sampler(
	parameterization = blob, 
	jacobian = J_det, 
	min_params = [-p, -p],
	max_params = [1+p, 1+p],
	max_jacobian = max_det
)

plot_image(sampler(1).reshape((17,17)), nc)

blob_images_uniform = sampler(1000)

# %% Get tallem results on uniform blobs in domain
import pickle
nc = 24.31768817109262 ## normalizing constant
X = pickle.load(open('/Users/mpiekenbrock/tallem/blob_images_uniform_domain.p', 'rb'))

## Try PH -- Nope!
from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(np.asanyarray(X))['dgms']
plot_diagrams(diagrams, show=True)
# plot_image(blob([0.5, 0.5]).reshape((17,17)), nc)

#%% Tallem on blobs uniform distributed in domain
import pickle
import numpy as np
from src.tallem import TALLEM
from src.tallem.cover import LandmarkCover

X = pickle.load(open('/Users/mpiekenbrock/tallem/blob_images_uniform_domain.p', 'rb'))
cover = LandmarkCover(X, 20).construct(X)
emb = TALLEM(cover, local_map="cmds2", n_components=3).fit_transform(X)

## Rotate embedding + color by distance to centroid
from src.tallem.color import linear_gradient, bin_color, colors_to_hex
col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

from src.tallem.distance import dist
centroid = emb.mean(axis=0)
dist_to_center = dist(centroid[np.newaxis, :], emb)

## Color poles green 
center_pole = np.argmin(np.array([np.linalg.norm(x - blob([0.5, 0.5])) for x in X]))
dark_pole = np.argmin(np.array([np.linalg.norm(x - blob([-0.50, -0.50])) for x in X]))

for angle in range(0, 360, 15):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*emb.T, color=bin_color(dist_to_center, col_pal)[0])
	ax.scatter3D(*emb[[center_pole, dark_pole],:].T, color=colors_to_hex(["green"])[0], s=100)
	ax.view_init(30, angle)
	plt.pause(0.75)

# %% Get tallem results on uniform blobs in codomain
import pickle
nc = 24.31768817109262 ## normalizing constant
X = pickle.load(open('/Users/mpiekenbrock/tallem/blob_images_uniform_codomain.p', 'rb'))

cover = LandmarkCover(X, 20).construct(X)
emb = TALLEM(cover, local_map="cmds2", n_components=3).fit_transform(X)

## Rotate embedding + color by distance to centroid
from src.tallem.color import linear_gradient, bin_color
col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

from src.tallem.distance import dist
centroid = emb.mean(axis=0)
dist_to_center = dist(centroid[np.newaxis, :], emb)

## Color poles green 
center_pole = np.argmin(np.array([np.linalg.norm(x - blob([0.5, 0.5])) for x in X]))
dark_pole = np.argmin(np.array([np.linalg.norm(x - blob([-0.50, -0.50])) for x in X]))

for angle in range(0, 360, 15):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*emb.T, color=bin_color(dist_to_center, col_pal)[0])
	ax.scatter3D(*emb[[center_pole, dark_pole],:].T, color=colors_to_hex(["green"])[0], s=100)
	ax.view_init(30, angle)
	plt.pause(0.75)


# %% Profile 
TALLEM(cover, local_map="cmds2", n_components=3)._profile(X=X, B=X)

# %% 
# pickle.dump(blob_images, open('blob_images_uniform_domain.p', 'wb'))
# pickle.dump(blob_images_uniform, open('blob_images_uniform_codomain.p', 'wb'))


# def mvn_density(x, mu, Sigma):
# 	x, mu = np.asanyarray(x), np.asanyarray(mu)
# 	if x.ndim == 1: np.array(x).reshape((len(x), 1))
# 	if mu.ndim == 1: np.array(mu).reshape((len(mu), 1))
# 	Sigma_inv = np.linalg.inv(Sigma)
# 	denom = np.sqrt(((2*np.pi)**2) * np.linalg.det(Sigma))
# 	return(np.exp((-0.5*((x - mu).T @ Sigma_inv @ (x - mu))))/denom)


#%% 
samples = []
pt_colors = []
for x in np.linspace(0.0-p,1.0+p,30):
	for y in np.linspace(0.0-p,1.0+p,30):
		samples.append(blob([x, y]))
		d = np.linalg.norm(np.array([x,y]) - np.array([0.5, 0.5]))
		pt_colors.append(d)
		# plot_image(blob([x, y]).reshape((17,17)), max_val=nc)

# NP = blob([0.5, 0.5])
# for t in np.linspace(0, 1, 100):
# 	samples.append(t*NP)
# 	pt_colors.append(1-t)

X = np.vstack(samples)	

# %% 
from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(X, maxdim=2)['dgms']
plot_diagrams(diagrams, show=True)

#%% 
from src.tallem import TALLEM 
from src.tallem.cover import LandmarkCover
cover = LandmarkCover(X, 8).construct(X)


from src.tallem.dimred import isomap
local_map = lambda x: isomap(x, d=3, k=15)

top = TALLEM(cover, local_map=local_map, n_components=3)
emb = top.fit_transform(X) # make work for X = distance matrices

top._profile(X=X, B=X)

#%% 
from src.tallem.color import linear_gradient, bin_color
col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

for angle in range(0, 360, 15):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*emb.T) #color=bin_color(np.array(pt_colors), col_pal)
	ax.view_init(30, angle)
	plt.pause(0.75)

# ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
# ax.scatter3D(*emb.T)

# %% 
from src.tallem.dimred import isomap
import matplotlib.pyplot as plt
Y = isomap(X, k = 5, d = 3)
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
ax.scatter3D(*Y.T)










# %% 
from src.tallem import TALLEM
from src.tallem.cover import IntervalCover, LandmarkCover
from src.tallem.datasets import mobius_band
import numpy as np

## Generate mobius band + polar coordinate 
M = mobius_band(n_polar=120, n_wide=15, scale_band = 0.25, plot=False, embed=6)
X, B = M['points'], M['parameters'][:,[1]]

## Use cyclic cover 
m_dist = lambda x,y: np.sum(np.minimum(abs(x - y), (2*np.pi) - abs(x - y)))
cover = IntervalCover(B, n_sets = 20, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)

## Run TALLEM
embedding = TALLEM(cover, local_map="pca2", n_components=2)
coords = embedding.fit_transform(X=X, B=B)

# %% Plot
import matplotlib.pyplot as plt
from src.tallem.color import bin_color, linear_gradient
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
ax.scatter3D(*coords.T, c = B)



# %% Neighborhood graph benchmarks
import numpy as np
from src.tallem.dimred import neighborhood_list, neighborhood_graph
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

X = np.random.uniform(size=(25, 2))

np.all(G1.A == G2.A)
G2.A[0,:]

# %% 
%%time 
G1 = neighborhood_graph(X, k = 16)



#%% 
%%time
G2 = kneighbors_graph(X, n_neighbors=15, mode="distance")


np.max(np.abs(G1.A - G2.A))
dist(X, as_matrix=True)[0,1]
np.all(G1.A == G2.A)
np.max(np.abs(G1.A[0,:] - G2.A[0,:]))

# %% 
%%time 
index = NNDescent(X, n_neighbors=15, compressed=False)
index.prepare()
G = index.neighbor_graph



#%% 
from fastdist import fastdist
from src.tallem.distance import dist

# %% 
%%time
wut = fastdist.matrix_pairwise_distance(X, fastdist.euclidean, "euclidean", return_matrix=False)

# %% 
%%time
wut = dist(X)

# from pynndescent import NNDescent, PyNNDescentTransformer

# import numpy as np

# X = np.random.uniform(size=(1000,10))

# transformer = PyNNDescentTransformer(n_neighbors=10)
# transformer.prepare()

# Check if package exists: 
# import importlib
# spam_spec = importlib.util.find_spec("spam") is not None

import numpy as np
X = np.random.uniform(size=(100,2))

from src.tallem.samplers import landmarks
from src.tallem.distance import dist
# landmarks(X, k = 15)

landmarks(X, k = 15)[1]
landmarks(dist(X, as_matrix=True), k = 15)[1]
landmarks(dist(X, as_matrix=False), k = 15)[1]

np.ravel(np.sqrt(landmarks(X, k = 15)[1]))
np.ravel(landmarks(dist(X, as_matrix=False), k = 15)[1])
np.ravel(landmarks(dist(X, as_matrix=True), k = 15)[1])
