# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# %% imports
from src.tallem.cover import IntervalCover
from src.tallem.dimred import neighborhood_graph
import numpy as np
import matplotlib.pyplot as plt
from src.tallem.landmark import landmarks
from src.tallem.utility import as_np_array
from src.tallem.distance import is_distance_matrix

a = np.random.uniform(size=(100,2))
b = a[landmarks(a, 20)['indices'],:]


# %% ball cover 
r = np.min(landmarks(a, 20)['radii'])
cover = BallCover(b, r)



#%% plot 
fig = plt.figure()
plt.scatter(a[:,0], a[:,1], c="blue")
plt.scatter(b[:,0], b[:,1], c="red")


# %% neighborhood code - 
# returns and adjacency list (as a n x m sparse matrix) giving k-nearest neighbor distances (or eps-ball distances)
# between the points in 'b' to the points in 'a'. If a == b, this is equivalent to computing the (sparse) neighborhood 
# graph as an adjacency matrix 


from src.tallem.dimred import neighborhood_list

# %% The final function! 
A = neighborhood_list(a,a,radius=0.15)


# %% Ball cover
radii = np.ravel(np.random.uniform(size=(b.shape[0],1)))
C = neighborhood_list(centers=b,a=a,radius=radii)


np.nonzero(C[:,5].todense())[0]



# %% Voronoi cover 

from scipy.spatial import Voronoi, voronoi_plot_2d
v = Voronoi(points=b)

#voronoi_plot_2d(v)


# %% KDtree cover 
from sklearn.neighbors import KDTree
a = np.random.uniform(size=(100,2))
tree = KDTree(a, leaf_size=5)
node_bnds = tree.get_arrays()[3]

from matplotlib.patches import Rectangle

leaf_status = np.array([is_leaf for istart, iend, is_leaf, radius in tree.get_arrays()[2]])
leaf_ind = np.where(leaf_status > 0)[0]

fig, ax = plt.subplots()
plt.scatter(a[:,0], a[:,1])
R = []
for i in leaf_ind:
	bnds = node_bnds[:,i,:]
	t = bnds.mean(axis=0)
	bnds = ((node_bnds[:,i,:]-t)*1.2)+t
	bl = bnds[0,:]
	tr = bnds[1,:]
	diff = np.abs(tr - bl)
	r = Rectangle(bl, width=diff[0], height=diff[1], fill=False, edgecolor="red")

	R.append(r)
	ax.add_patch(r)
plt.show()

# %% Box landmarks
from src.tallem.landmark import landmarks
box_landmarks = landmarks(a, 16, metric = "chebychev")

fig, ax = plt.subplots()
plt.scatter(a[:,0], a[:,1])
R = []
d_min = np.min(box_landmarks['radii'])
for i, d in zip(box_landmarks['indices'], box_landmarks['radii']):
	if i == 0: continue
	d = d_min
	bl = a[i,:] - np.array([d, d])
	r = Rectangle(bl, width=2*d, height=2*d, fill=False, edgecolor="red")
	R.append(r)
	ax.add_patch(r)
plt.show()

# %% Meanshift cover 
import numpy as np
from scipy.stats import gaussian_kde
a = np.random.uniform(size=(100,2))

kde = gaussian_kde(a.T)

x = a[0,:]
m = lambda x: next((kx * x[:,np.newaxis])/np.sum(kx) for kx in [kde((a - x).T)[np.newaxis,:]])
# np.sum(kde((a - x).T))


# %% Quantized cover
from scipy.cluster.vq import kmeans2
from scipy.spatial import Voronoi, voronoi_plot_2d
centroids, cl = kmeans2(a, k=5, minit='++')
vor = Voronoi(centroids)

voronoi_plot_2d(vor)

import shapely.geometry
import shapely.ops

lines = [
	shapely.geometry.LineString(vor.vertices[line])
	for line in vor.ridge_vertices
	if -1 not in line
]
polys = shapely.ops.polygonize(lines) # only interior bounded polygons 

for p in polys:
	print(p)
# G = neighborhood_list(centroids, a, k=1)
# G.nonzero()[1]

fig, ax = plt.subplots()
plt.scatter(a[:,0], a[:,1])
plt.scatter(centroids[:,0], centroids[:,1], color="red")


# %%
import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import scipy.spatial
import sys

eps = sys.float_info.epsilon

n_towers = 100
towers = np.random.rand(n_towers, 2)
bbox = np.array([0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]

def in_box(a: npt.ArrayLike, bbox: npt.ArrayLike):
	return((a[:,0] >= bbox[0]) & (a[:,0] <= bbox[1]) & (a[:,1] >= bbox[2]) & (a[:,1] <= bbox[3]))

def voronoi(towers, bbox):
	# Select towers inside the bounding box
	i = in_box(towers, bbox)
	# Mirror points
	points_center = towers[i, :]
	points_left = np.c_[bbox[0] - (points_center[:,0] - bbox[0]), points_center[:,1]]
	points_right = np.c_[bbox[1] + (bbox[1] - points_center[:, 0]), points_center[:,1]]
	points_down = np.c_[points_center[:,0], bbox[2] - (points_center[:,1] - bbox[2])]
	points_up = np.c_[points_center[:,0], bbox[3] + (bbox[3] - points_center[:,1])]
	points = np.vstack((points_center, points_left, points_right, points_down, points_up))

	# Compute Voronoi
	vor = sp.spatial.Voronoi(points)
	# Filter regions
	regions = []
	for region in vor.regions:
		flag = True
		for index in region:
			if index == -1:
				flag = False
				break
			else:
				x, y = vor.vertices[index, 0], vor.vertices[index, 1]
				in_x = bbox[0] - eps <= x and x <= bbox[1] + eps
				in_y = bbox[2] - eps <= y and y <= bbox[3] + eps
				if not(in_x and in_y):
					flag = False
					break
		if region != [] and flag:
			regions.append(region)
	vor.filtered_points = points_center
	vor.filtered_regions = regions
	return vor

vor = voronoi(towers, bounding_box)
v_cells = [vor.vertices[region] for region in vor.filtered_regions]

v_cover = []
for cell in v_cells:
	centroid = np.mean(cell, axis = 0)
	q = (1.3*(cell - centroid))+centroid
	v_cover.append(q)

fig = pl.figure()
ax = fig.gca()
ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
for p in v_cover:
	p = np.vstack((p, p[0,:]))
	ax.plot(p[:,0], p[:,1], 'k-')

fig = pl.figure()
ax = fig.gca()
# Plot initial points
ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
for region in vor.filtered_regions:
    vertices = vor.vertices[region + [region[0]], :]
    ax.plot(vertices[:, 0], vertices[:, 1], 'k-')

# Compute and plot centroids
centroids = []
for region in vor.filtered_regions:
    vertices = vor.vertices[region + [region[0]], :]
    centroid = centroid_region(vertices)
    centroids.append(list(centroid[0, :]))
    ax.plot(centroid[:, 0], centroid[:, 1], 'r.')

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
pl.savefig("bounded_voronoi.png")

sp.spatial.voronoi_plot_2d(vor)
pl.savefig("voronoi.png")

