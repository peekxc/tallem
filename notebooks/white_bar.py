# %% Show grid of samples
import numpy as np
from tallem.datasets import *

bar, c = white_bars(n_pixels=25, r=0.23, sigma=2.25)
samples = []
for d in np.linspace(-0.5, 0.5, num=9, endpoint=True):
	for theta in np.linspace(0, np.pi, num=11, endpoint=True):
		samples.append(np.ravel(bar(theta, d)).flatten())
samples = np.vstack(samples)
fig, ax = plot_images(samples, shape=(25,25), max_val=c, layout=(9,11))

# %% Oversample  + landmarks to get a uniform sampling
n_params = 50
X, B = np.zeros((n_params**2, 25**2)), np.zeros((n_params**2,2))
cc = 0
for d in np.linspace(-0.5, 0.5, num=n_params, endpoint=True):
	for theta in np.linspace(0, np.pi, num=n_params, endpoint=True):
		X[cc,:] = np.ravel(bar(theta, d)).flatten()
		B[cc,:] = np.array([theta, d])
		cc += 1

## Choose landmarks using the intrinsic metric 
from tallem.samplers import landmarks
Lind, Lrad = landmarks(B, 1200)
XL = X[Lind,:]
BL = B[Lind,:]

# %% Show parameter space
scatter2D(BL, c=BL[:,0], figsize=(5,3))

# %% Tallem on true polar coordinate
from tallem import TALLEM
from tallem.cover import CircleCover
polar_coordinate = BL[:,0]
cover = CircleCover(polar_coordinate, n_sets=15, scale=1.5, lb=0, ub=np.pi) # 20, 1.5 is best
top = TALLEM(cover, local_map="pca2", D=3, pou="triangular").fit(X=XL)
print(top)

# %% Nerve complex 
top.plot_nerve(X=XL, layout="hausdorff")

# %% Embedding colored by polar coordinate
emb = top.fit_transform(X=XL)
fig, ax = scatter3D(emb, c=polar_coordinate)

# %% Embedding colored by signed distance 
fig, ax = scatter3D(emb, c=BL[:,1])


# %% Look at images in each open 
ind = np.random.choice(top.cover[0], size=50)
fig, ax = plot_images(XL[ind,:], shape=(25,25), max_val=c, layout=(5,10), figsize=(8,4))

# %% 
ind = np.random.choice(top.cover[14], size=50)
fig, ax = plot_images(XL[ind,:], shape=(25,25), max_val=c, layout=(5,10), figsize=(8,4))

# %%
ind = np.random.choice(top.cover[7], size=50)
fig, ax = plot_images(XL[ind,:], shape=(25,25), max_val=c, layout=(5,10), figsize=(8,4))

# %% 
ind = np.random.choice(top.cover[13], size=50)
fig, ax = plot_images(XL[ind,:], shape=(25,25), max_val=c, layout=(5,10), figsize=(8,4))


# %% 
from tallem import TALLEM
from tallem.cover import LandmarkCover
cover = LandmarkCover(XL, n_sets=15, scale=1.5)
top = TALLEM(cover, local_map="pca2", D=3, pou="triangular").fit(X=XL)
print(top)

# %% Nerve complex 
top.plot_nerve(X=XL, layout="hausdorff")

# %% 
fig, ax = scatter3D(top.embedding_, c=BL[:,0])


# %% Isomap 
from tallem.distance import dist
from tallem.dimred import isomap, connected_radius, enclosing_radius

pc, pe = connected_radius(dist(XL, as_matrix=True)), enclosing_radius(dist(XL, as_matrix=True))

all_isos = [isomap(XL, d=3, r=p) for p in np.linspace(pc, pe, 30)]
	
fig, ax = scatter3D(, c=BL[:,0])

