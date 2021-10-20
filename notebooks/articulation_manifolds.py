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
embedding = TALLEM(cover, local_map="pca2", n_components=3).fit_transform(X, polar_coordinate)

## Rotate and view
angles = np.linspace(0, 360, num=12, endpoint=False)
scatter3D(embedding, angles=angles, figsize=(16, 8), layout=(2,6), c=polar_coordinate)


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

cover = LandmarkCover(Disks, k=8, scale=1.8)
# local_map = lambda x: pca(x, d=1)
top = TALLEM(cover, local_map="pca2", n_components=2)
emb = top.fit_transform(X=Disks, B=Disks)
P = [
	cmds(Disks, d=2),
	isomap(Disks, d=2, k=10), 
	isomap(Disks, d=2, p=0.15), 
	emb
]
scatter2D(P, layout=(1,4), figsize=(8,3), c = Theta)

# local_models = [m for m in top.models.values()]
# scatter2D(local_models, layout=(1,len(local_models)), figsize=(8,3))

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
cover = LandmarkCover(samples, k=20, scale=1.8)
assert(np.all(np.array([len(s) for s in cover.values()]) > 1))

top = TALLEM(cover, local_map="cmds3", n_components=3)
emb = top.fit_transform(X=samples, B=samples)

## Eccentricity for color
ecc = np.array([np.linalg.norm(p - np.array([0.5, 0.5, 0.0])) for p in params])
angles = np.linspace(0, 360, num=6, endpoint=False)
scatter3D(emb, c=ecc, angles=angles, layout=(2, 3), figsize=(18,12))

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

cover = LandmarkCover(bars, k=15, scale=1.5)
assert np.all(np.array([len(s) for s in cover.values()]) > 1)
assert validate_cover(bars.shape[0], cover)

top = TALLEM(cover, local_map="cmds3", n_components=3)
emb = top.fit_transform(X=bars, B=bars)

## Use parameters for color
angles = np.linspace(0, 360, num=6, endpoint=False)
scatter3D(emb, c=params[:,0], angles=angles, layout=(2, 3), figsize=(14,10))

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






