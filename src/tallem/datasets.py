import numpy as np
from matplotlib.tri import Triangulation
import matplotlib.pyplot as pyplot

def flywing():
	''' Fly wings example (Klingenberg, 2015 | https://en.wikipedia.org/wiki/Procrustes_analysis) '''
	arr1 = np.array([[588.0, 443.0], [178.0, 443.0], [56.0, 436.0], [50.0, 376.0], [129.0, 360.0], [15.0, 342.0], [92.0, 293.0], [79.0, 269.0], [276.0, 295.0], [281.0, 331.0], [785.0, 260.0], [754.0, 174.0], [405.0, 233.0], [386.0, 167.0], [466.0, 59.0]])
	arr2 = np.array([[477.0, 557.0], [130.129, 374.307], [52.0, 334.0], [67.662, 306.953], [111.916, 323.0], [55.119, 275.854], [107.935, 277.723], [101.899, 259.73], [175.0, 329.0], [171.0, 345.0], [589.0, 527.0], [591.0, 468.0], [299.0, 363.0], [306.0, 317.0], [406.0, 288.0]])
	return([arr1, arr2])

def mobius_band(n_polar=66, n_wide=9, scale_band=0.25, embed=3, plot=False):
	# %% Generate small data set on Mobius Band 
	s = np.linspace(-scale_band, scale_band, 2*n_wide)	# width/radius
	t = np.linspace(0, 2*np.pi, n_polar)   # circular coordinate 
	s, t = np.meshgrid(s, t)

	# radius in x-y plane
	phi = 0.5 * t
	r = 1 + s * np.cos(phi)
	x = np.ravel(r * np.cos(t))
	y = np.ravel(r * np.sin(t))
	z = np.ravel(s * np.sin(phi))
	tri = Triangulation(np.ravel(s), np.ravel(t))

	# %% Stratify sample from triangulation using barycentric coordinates 
	mobius_sample = []
	polar_sample = []
	S, T = np.ravel(s), np.ravel(t)
	for i in range(tri.triangles.shape[0]):
		weights = np.random.uniform(size=3)
		weights /= np.sum(weights)
		pts = np.ravel([(t[0], t[1], t[2]) for t in [tri.triangles[i]]])
		xyz = [np.sum(weights*x[pts]), np.sum(weights*y[pts]), np.sum(weights*z[pts])]
		mobius_sample.append(xyz)
		polar_sample.append([np.sum(weights*S[pts]), np.sum(weights*T[pts])])
	mobius_sample = np.array(mobius_sample)
	polar_sample = np.array(polar_sample)

	# Plot mobius band sample 
	if plot:
		ax = pyplot.axes(projection='3d')
		ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap='viridis', linewidths=0.2, alpha=0.20)
		ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1);
		ax.scatter3D(mobius_sample[:,0], mobius_sample[:,1], mobius_sample[:,2], c="red",s=0.20)

	# Embed in higher dimensions
	if embed > 3:
		D = embed
		def givens(i,j,theta,n=2):
			G = np.eye(n)
			G[i,i] = np.cos(theta)
			G[j,j] = np.cos(theta)
			G[i,j] = -np.sin(theta)
			G[j,i] = np.sin(theta)
			return(G)

		## Append zero columns up to dimension d
		M = np.hstack((mobius_sample, np.zeros((mobius_sample.shape[0], D - mobius_sample.shape[1]))))

		## Rotate into D-dimensions
		from itertools import combinations
		for (i,j) in combinations(range(6), 2):
			theta = np.random.uniform(0, 2*np.pi)
			G = givens(i,j,theta,n=D)
			M = M @ G
	else: 
		M = mobius_sample
		
	# Return sample points + their polar/width parameters
	return({ "points" : M, "parameters": polar_sample })


# # %% Visualize isomap 
# from tallem.isomap import isomap
# embedded_isomap = isomap(M, d = 3, k=5)
# ax = pyplot.axes(projection='3d')
# ax.scatter3D(embedded_isomap[:,0], embedded_isomap[:,1], embedded_isomap[:,2], c="red",s=0.20)

# # %% Sklearn isomap to verify
# from sklearn.manifold import Isomap
# embedded_isomap = Isomap(n_components=3, n_neighbors=5).fit_transform(M)
# ax = pyplot.axes(projection='3d')
# ax.scatter3D(embedded_isomap[:,0], embedded_isomap[:,1], embedded_isomap[:,2], c="red",s=0.20)

# # %% MDS 
# from tallem.mds import classical_MDS
# from tallem.distance import dist
# embedded_cmds = classical_MDS(dist(M, as_matrix=True), k = 3)
# ax = pyplot.axes(projection='3d')
# ax.scatter3D(embedded_cmds[:,0], embedded_cmds[:,1], embedded_cmds[:,2], c="red",s=0.20)

# # %% Distortion
# dist_truth = dist(mobius_sample)
# print(np.linalg.norm(dist(M) - dist_truth))
# print(np.linalg.norm(dist(embedded_isomap) - dist_truth))
# print(np.linalg.norm(dist(embedded_cmds) - dist_truth))
