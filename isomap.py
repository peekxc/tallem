# %% Isomap imports
import numpy as np
import numpy.typing as npt
from distance import dist 
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix
from mds import sammon, classical_MDS
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components

# %% Isomap Definitions
def neighborhood_graph(a: npt.ArrayLike, radius: float = None, k: int = None, **kwargs):
	''' 
	Computes the neighborhood graph of a point cloud. 
	Returns a sparse weighted adjacency matrix where positive entries indicate the distance between points in X 
	'''
	if radius is None and k is None: raise RuntimeError("Either radius or k must be supplied")
	a = np.asarray(a)
	n = a.shape[0]

	## If 'a' is any array, form a KD tree to extract the neighbors
	tree = KDTree(data=a, **kwargs)
	if radius is not None:
		pairs = tree.query_pairs(r=radius)
		r = [p[0] for p in pairs]
		c = [p[1] for p in pairs]
		d = dist(a[r,:], a[c,:], pairwise = True)
	else:
		knn = tree.query(a, k=k+1)
		r = np.repeat(range(n), repeats=k)
		c = knn[1][:,1:].flatten()
		d = knn[0][:,1:].flatten()
	
	## Form the neighborhood graph 
	D = csc_matrix((d, (r, c)), dtype=np.float32, shape=(n,n))
	return(D)

def floyd_warshall(a: npt.ArrayLike):
	'''floyd_warshall(adjacency_matrix) -> shortest_path_distance_matrix
	Input
			An NxN NumPy array describing the directed distances between N nodes.
			adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)
			Notes:
			* If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
			* The diagonal of adjacency_matrix should be zero.
			Based on https://gist.github.com/mosco/11178777
	Output
			An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
	'''
	n = a.shape[0]
	if not(a[0,0] == 0.0): np.fill_diagonal(a, 0.0) # quick check to make sure diagonal is 0
	for k in range(n):
			a = np.minimum(a, a[np.newaxis,k,:] + a[:,k,np.newaxis])
	return a

def radius_bounds(a: npt.ArrayLike):
	''' 
	Given a distance matrix, returns an interval indicating a possible range 
	to create a ball neighborhood graph over. The lower bound of the range is the 
	smallest radii such that the space is connected and the upper bound of the 
	range is the radius that renders the space contractible to a point. 
	'''
	min_radii = np.max(minimum_spanning_tree(a))
	max_radii = np.min(np.amax(a, axis = 0))
	return((min_radii, max_radii))
 
def isomap(a: npt.ArrayLike, d: int, coord: str = "mMDS", **kwargs):
	a = np.asarray(a)
	# if a.shape[0] != a.shape[1]: raise ValueError("Input matrix must be a square distance matrix.")
	# if not all(np.diag(a) == 0): raise ValueError("Input matrix must be a square distance matrix.")
	G = neighborhood_graph(a, **kwargs)
	if not(connected_components(G, directed = False)[0] == 1):
		raise RuntimeError("Can only run isomap on a connected neighborhood graph")
	D = np.ascontiguousarray(G.todense())
	D = np.maximum(D, D.T) # symmetrize
	D[np.where(D == 0)] = np.inf
	np.fill_diagonal(D, 0.0)
	D = floyd_warshall(D)
	if np.any(D == np.inf): raise RuntimeError("Failed to form finite geodesics for every pair of points.")
	if coord == "mMDS":
		X = classical_MDS(D, k=d, coords=True)
	elif coord == "sammon":
		X = sammon(D, k=d)[0]
	return(X)

def partition_of_unity(a: npt.ArrayLike, centers: npt.ArrayLike, f = "tent"):
	n = a.shape[0]
	J = centers.shape[0]
	P = np.zeros(shape = (J, n), dtype = np.float32)
	for j in range(J):
		for i in range(n):
				P[j,i] = dist()	
	dist()
	



## TODO: parallel coordinates plot
from scipy.sparse.csgraph import floyd_warshall as fw

# %%
X = np.random.uniform(size=(15,2))
isomap(X, d = 2, k = 3)


# n = X.shape[0]
# simplex 
# D = -np.eye(4) + np.ones(shape=(4,4))


# %%
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.draw()
ax = plt.axes(projection='3d')
x = classical_MDS(D, k = 3, coords = True)
ax.scatter3D(x[:,0], x[:,1], x[:,2])
plt.show()

# %%
from sklearn.manifold import MDS
embed = MDS(n_components=2, dissimilarity='euclidean')
dist(embed.fit_transform(D), as_matrix=True)
