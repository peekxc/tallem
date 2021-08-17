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

#%% plot 
fig = plt.figure()
plt.scatter(a[:,0], a[:,1], c="blue")
plt.scatter(b[:,0], b[:,1], c="red")


# %% neighborhood code - 
# returns and adjacency list (as a n x m sparse matrix) giving k-nearest neighbor distances (or eps-ball distances)
# between the points in 'b' to the points in 'a'. If a == b, this is equivalent to computing the (sparse) neighborhood 
# graph as an adjacency matrix 
from scipy.sparse import lil_matrix
metric = "euclidean"
k = 15
a, b = as_np_array(a), as_np_array(b)
n, m = a.shape[0], b.shape[0]
minkowski_metrics = ["cityblock", "euclidean", "chebychev"]

G = lil_matrix((n, m))

if not(is_distance_matrix(a)) and not(is_distance_matrix(b)): 
	if metric in minkowski_metrics:
		p = [1, 2, float("inf")][minkowski_metrics.index(metric)]
		tree = KDTree(data=a, **kwargs)
		tree = KDTree(data=b)
		if radius is not None:

			for j, nn_idx in enumerate(tree.query_ball_point(a, r=radius, p=p)):
				G[nn_idx,j] = 
			pairs = tree.query_pairs(r=radius*2.0)
			r, c = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
			d = dist(a[r,:], a[c,:], pairwise = True)
		else:
			knn = tree.query(a, k=k+1)
			r, c, d = np.repeat(range(a.shape[0]), repeats=k), knn[1][:,1:].flatten(), knn[0][:,1:].flatten()
	else: 
	
	## If 'a' is a point cloud, form a KD tree to extract the neighbors
	if not(is_distance_matrix(a)) and not(is_distance_matrix(b)):
		tree = KDTree(data=a, **kwargs)
		
	else: 
		if radius is not None: 
			r, c = np.where(a <= (radius*2.0))
			valid = (r != c) & (r < c)
			r, c = r[valid], c[valid]
			d = a[r,c]
		else: 
			knn = np.apply_along_axis(lambda a_row: np.argsort(a_row)[0:(k+1)],axis=1,arr=a)
			r, c = np.repeat(range(n), repeats=5), np.ravel(knn[:,1:])
			d = a[r,c]

	## Form the neighborhood graph 
	D = csc_matrix((d, (r, c)), dtype=np.float32, shape=(n,n))
	return(D)

