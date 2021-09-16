# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

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

import autograd.numpy as auto_np
from autograd import jacobian
scale, image_sz, s = 1, (17, 17), 1
def gaussian_pixel2(center):
	x, y = auto_np.meshgrid(range(image_sz[0]), range(image_sz[0]))
	#grid = (auto_np.abs(center[0] - x)**2 + auto_np.abs(center[1] - y))**2
	grid = auto_np.exp(-0.5*((x - center[0])**2 + (y - center[1])**2))/auto_np.sqrt((2*auto_np.pi)**2)
	#grid = auto_np.zeros(shape=image_sz)	# ind = auto_np.ix_(range(image_sz[0]), range(image_sz[1]))
	#grid[auto_np.ix_(range(image_sz[0]), range(image_sz[0]))] = auto_np.sum(center**2)
	# for i in range(image_sz[0]):
	# 	for j in range(image_sz[1]):
	# 		grid[i,j] = auto_np.exp(-0.5*((i - center[0])**2 + (j - center[1])**2))/auto_np.sqrt((2*auto_np.pi)**2)
	return(np.ravel(grid)

# jacobian(gaussian_pixel2)(auto_np.array([0.,0.]))


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

phi = np.random.uniform(low=0, high=2*np.pi, size=1000)
theta = np.random.uniform(low=0, high=2*np.pi, size=1000)
T = np.asanyarray([f(x) for x in np.c_[phi, theta]])
seaborn.pairplot(pandas.DataFrame(T))

## Uniformly sampled torus using autograd 
J_det = J_determinant(f)

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
T2 = sampler(1000)
seaborn.pairplot(pandas.DataFrame(T2))

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


## WRONG
np.linalg.norm(gaussian_pixel2(auto_np.array([0.1,0.1])) - gaussian_pixel2(auto_np.array([0.1,0.1])))

sampler = rejection_sampler(
	parameterization = f, 
	jacobian = J_det, 
	min_params = [-1.0, -1.0],
	max_params = [1.0, 1.0],
	max_jacobian = 1.0
)





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






