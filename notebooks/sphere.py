# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))


#%% Generate uniform points
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
H = Hypersphere(2) # hs = HypersphereMetric(2)
p_ext = H.random_uniform(1500)
# p_int = H.extrinsic_to_intrinsic_coords(p_ext) # both coordinates between [-1, 1]
# p_int = (p_int + 1.0)/2.0

ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
ax.scatter3D(*p_ext.T, c=dist_to_center)

# %% 
from src.tallem import TALLEM
from src.tallem.cover import LandmarkCover

cover = LandmarkCover(p_ext, 25, scale = 1.5)
pc = TALLEM(cover, local_map="iso2", n_components=3).fit_transform(X=p_ext,B=p_ext)


import numpy as np
mu = p_ext.mean(axis=0)
dist_to_center = np.array([np.linalg.norm(p - mu) for p in p_ext])

import matplotlib.pyplot as plt 
for angle in range(0, 360, 30):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*pc.T, c=dist_to_center)
	ax.view_init(30, angle)
	plt.pause(0.50)


# %% 
from src.tallem import TALLEM
from src.tallem.cover import LandmarkCover

from src.tallem.dimred import geodesic_dist, neighborhood_graph
G = neighborhood_graph(p_ext, k = 15)
D = geodesic_dist(G.A)

cover = LandmarkCover(D, 25, scale = 1.5)
pc = TALLEM(cover, local_map="iso2", n_components=3).fit_transform(X=p_ext,B=p_ext)


import numpy as np
mu = p_ext.mean(axis=0)
dist_to_center = np.array([np.linalg.norm(p - mu) for p in p_ext])

import matplotlib.pyplot as plt 
for angle in range(0, 360, 30):
	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
	ax.scatter3D(*pc.T, c=dist_to_center)
	ax.view_init(30, angle)
	plt.pause(0.50)

# from src.tallem.color import linear_gradient, bin_color
# col_pal = linear_gradient(["red", "purple", "blue"], 25)['hex']

# for angle in range(0, 360, 30):
# 	ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
# 	ax.scatter3D(*pc.T, color=bin_color(knn_dist, col_pal))
# 	ax.view_init(30, angle)
# 	plt.pause(1.50)
