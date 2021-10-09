import numpy as np
from typing import *
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

def flywing():
	''' Fly wings example (Klingenberg, 2015 | https://en.wikipedia.org/wiki/Procrustes_analysis) '''
	arr1 = np.array([[588.0, 443.0], [178.0, 443.0], [56.0, 436.0], [50.0, 376.0], [129.0, 360.0], [15.0, 342.0], [92.0, 293.0], [79.0, 269.0], [276.0, 295.0], [281.0, 331.0], [785.0, 260.0], [754.0, 174.0], [405.0, 233.0], [386.0, 167.0], [466.0, 59.0]])
	arr2 = np.array([[477.0, 557.0], [130.129, 374.307], [52.0, 334.0], [67.662, 306.953], [111.916, 323.0], [55.119, 275.854], [107.935, 277.723], [101.899, 259.73], [175.0, 329.0], [171.0, 345.0], [589.0, 527.0], [591.0, 468.0], [299.0, 363.0], [306.0, 317.0], [406.0, 288.0]])
	return([arr1, arr2])

def gaussian_blob(n_pixels: int, r: float):
	'''
	Generates a closure which, given a 2D location *mu=(x,y)*, generates a white blob 
	with [normalized] radius 0 < r <= 1 in a (n_pixels x n_pixels) image. 

	If *mu* is in [0,1] x [0,1], the center of the white blob should be visible
	If *mu* has as both of its coordinates outside of [0,1]x[0,1], the blob may be partially visible
	If *mu* has both of its coordinates outside of [-r, 1+r]x[-r, 1+r], then image should be essentially black

	The returned closure completely autograd's numpy wrapper to do the image generation. Thus, the resulting 
	function can be differentiated (w.r.t *mu*) using the reverse-mode differentiation process that *autograd* provides.

	This function also returns the global normalizing constant needed normalize the pixel intensities in [0,1],
	for plotting or other purposes.

	Return: (blob, c) where
	 - blob := differentiable closure which, given a vector (x,y), generates the blob image a flat vector.
	 - c := maximum value of the intensity of any given pixel for any choice of *mu*.
	'''
	import autograd.numpy as auto_np
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

# def _gaussian_pixel(d, n_pixels):
# 	from scipy.stats import norm
# 	sigma = d/3.0
# 	Sigma = auto_np.diag([sigma, sigma])
# 	sigma_inv = auto_np.linalg.inv(Sigma)[0,0]
# 	denom = np.sqrt(((2*np.pi)**2) * auto_np.linalg.det(Sigma))
# 	normal_constant = norm.pdf(0, loc=0, scale=sigma)
# 	def blob(mu): # generates blob at location mu 
# 		# mu = mu.reshape((2, 1))
# 		# np.exp(-0.5 * ((x - mu).T @ SigmaI @ (x - mu))).flatten()
# 		#x, y = auto_np.meshgrid(auto_np.arange(n_pixels), auto_np.arange(n_pixels))
# 		loc = auto_np.linspace(0, 1, n_pixels, False) + (1/(2*n_pixels))
# 		x,y = auto_np.meshgrid(loc, loc)
# 		grid = auto_np.exp(-0.5*(sigma_inv * ((x-mu[0])**2 + (y-mu[1])**2)))/denom
# 		#grid = auto_np.exp(-0.5*((x - mu[0])**2 + (y - mu[1])**2))/denom
# 		#return(auto_np.ravel(grid).flatten())
# 		return(grid/normal_constant)
# 	return(blob)
# plot_image(gaussian_pixel2(1/32, 11)([-0.5, 0.5]))


def white_dot(n_pixels: int, r: float, n: Optional[int], method: Optional[str] = "grid", mu: Optional[ArrayLike] = None):
	''' 
	Generates a grayscale image data set where white blobs are placed on a (n_pixels x n_pixels) grid
	using a multivariate normal density whose standard deviation sigma (in both directions) is sigma=d/3.
	If 'n' is specified, then 'n' samples are generated from a larger space s([-d, 1+d]^2) where s(*)
	denotes the scaling of the interval [-d,1+d] by 'n_pixels'. 
	'''
	assert r > 0 and r <= 1.0, "r must be in the range 0 < r <= 1.0"

	## First generate the closure to make the images
	blob, c = gaussian_blob(n_pixels, r)

	if not(mu is None):
		output = np.vstack([blob(auto_np.array([x,y])) for x,y in mu])
	elif method == "random":
		## Generate uniformly random locations (in domain)
		assert n is not None, "'n' must be supplied if 'mu' is not."
		X, Y = np.random.uniform(low=-r,high=1+r,size=n), np.random.uniform(low=-d,high=1+d,size=n)
		output = np.vstack([blob(auto_np.array([x,y])) for x,y in zip(X, Y)])
		params = np.c_[X, Y]
	elif method == "grid":
		assert n is not None, "'n' must be supplied if 'mu' is not."
		n1, n2 = (n, n) if isinstance(n, int) else (n[0], n[1])
		samples, params = [], []
		for x in np.linspace(0.0-r,1.0+r,n1):
			for y in np.linspace(0.0-r,1.0+r,n1):
				samples.append(blob(auto_np.array([x, y])))
				params.append([x, y, 1.0])
		
		## Generate the pole
		NP = blob(auto_np.array([0.5, 0.5]))
		for t in np.linspace(0, 1, n2):
			samples.append(t*NP)
			params.append([0.5, 0.5, 1-t])

		## Vertically stack 
		samples, params = np.vstack(samples), np.vstack(params)

	## Return the data 
	return(samples, params, blob, c)

def mobius_band(n_polar=66, n_wide=9, scale_band=0.25):
	''' Generates samples on a Mobius band embedded in R^3 '''

	## Generate random (deterministic) polar coordinates around Mobius Band
	np.random.seed(0) 
	s = np.linspace(-scale_band, scale_band, 2*n_wide)	# width/radius
	t = np.linspace(0, 2*np.pi, n_polar)   							# circular coordinate 
	s, t = np.meshgrid(s, t)

	## Triangulate to allow stratification
	M = np.c_[np.ravel(s), np.ravel(t)]
	V = M[Delaunay(M).simplices]

	## Sample within each strata via random barycentric coordinates 
	normalize = lambda x: x / np.sum(x) 
	Y = np.array([np.sum(v * normalize(np.random.uniform(size=(3,1))), axis = 0) for v in V])

	## Convert to 3d
	S, T = Y[:,0], Y[:,1]
	phi = 0.5 * T
	r = 1 + S * np.cos(phi)
	MB = np.c_[r * np.cos(T), r * np.sin(T), S * np.sin(phi)]

	## Return both 3D embedding + original parameters
	return({ "points" : MB, "parameters": Y })


def embed(a: ArrayLike, D: int, method="givens"):
	''' Embeds a point cloud into D dimensions using random orthogonal rotations '''
	def givens(i,j,theta,n=2):
		G = np.eye(n)
		G[i,i] = np.cos(theta)
		G[j,j] = np.cos(theta)
		G[i,j] = -np.sin(theta)
		G[j,i] = np.sin(theta)
		return(G)

	## Append zero columns up to dimension d
	d = a.shape[1]
	a = np.hstack((a, np.zeros((mobius_sample.shape[0], D - d))))

	## Rotate into D-dimensions
	from itertools import combinations
	for (i,j) in combinations(range(D), 2):
		theta = np.random.uniform(0, 2*np.pi)
		G = givens(i,j,theta,n=D)
		a = a @ G
	return(a)

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
