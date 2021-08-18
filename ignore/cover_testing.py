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

