import numpy as np
from typing import *
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from tallem.utility import ask_package_install, package_exists

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

def plot_image(P, figsize=(8,8), max_val = "default"):
	if max_val == "default": max_val = np.max(P)
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=figsize)
	plt.imshow(P, cmap='gray', vmin=0, vmax=max_val)
	fig.gca().axes.get_xaxis().set_visible(False)
	fig.gca().axes.get_yaxis().set_visible(False)

def plot_images(P, shape, max_val = "default", figsize=(8,8), layout = None):
	''' 
		P := numpy array where each row is a grayscale image
		shape := the shape to reshape each row of P prior to plotting
	'''
	import matplotlib.pyplot as plt
	if max_val == "default": 
		max_val = np.max(P)
	if P.ndim == 1:
		fig = plt.figure(figsize=figsize)
		plt.imshow(P.reshape(shape), cmap='gray', vmin=0, vmax=max_val)
		fig.gca().axes.get_xaxis().set_visible(False)
		fig.gca().axes.get_yaxis().set_visible(False)
		return(fig, fig.gca())
	else:
		assert layout is not None, "missing layout"
		fig, axs = plt.subplots(*layout, figsize=figsize)
		axs = axs.flatten()
		for i, (img, ax) in enumerate(zip(P, axs)):
			#fig.add_subplot(layout[0], layout[1], i+1)
			plt.axis("off")
			ax.imshow(P[i,:].reshape(shape), cmap='gray', vmin=0, vmax=max_val, aspect='auto')
			ax.axes.xaxis.set_visible(False)
			ax.axes.yaxis.set_visible(False)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
		return(fig, axs)

def scatter2D(P, layout = None, figsize=(8,8), **kwargs):
	import matplotlib.pyplot as plt
	if isinstance(P, np.ndarray):
		if "fig" in kwargs.keys() and "ax" in kwargs.keys():
			fig, ax = kwargs["fig"], kwargs["ax"]
			kwargs.pop('fig', None)
			kwargs.pop('ax', None)
		else:
			fig = plt.figure(figsize=figsize)
			ax = fig.add_subplot()
		ax.scatter(*P.T, **kwargs)
		return(fig, ax)
	elif isinstance(P, Iterable):
		assert layout is not None, "missing layout"
		assert len(P) == np.prod(layout)
		if "fig" in kwargs.keys() and "ax" in kwargs.keys():
			fig, ax = kwargs["fig"], kwargs["ax"]
			kwargs.pop('fig', None)
		else:
			fig = plt.figure(figsize=figsize)
		for i, p in enumerate(P):
			ax = fig.add_subplot(layout[0], layout[1], i+1)
			ax.scatter(*p.T, **kwargs) 
		return(fig, ax)

def scatter3D(P, angles = None, layout = None, figsize=(8,8), **kwargs):
	import matplotlib.pyplot as plt
	if isinstance(P, np.ndarray):
		import numbers
		if angles is not None:
			if isinstance(angles, numbers.Integral): 
				angles = np.linspace(0, 360, angles, endpoint=False)
			assert len(angles) == np.prod(layout)
			if "fig" in kwargs.keys() and "ax" in kwargs.keys():
				fig, ax = kwargs["fig"], kwargs["ax"]
				kwargs.pop('fig', None)
				kwargs.pop('ax', None)
			else: 
				fig, ax = plt.subplots(*layout, figsize=figsize)
			for i, theta in enumerate(angles):
				ax = fig.add_subplot(layout[0], layout[1], i+1, projection='3d')
				ax.scatter3D(*P.T, **kwargs) 
				ax.view_init(30, theta)
		else: 
			if "fig" in kwargs.keys() and "ax" in kwargs.keys():
				fig, ax = kwargs["fig"], kwargs["ax"]
				kwargs.pop('fig', None)
				kwargs.pop('ax', None)
			else: 
				fig = plt.figure(figsize=figsize)
				ax = fig.add_subplot(projection='3d')
			ax.scatter3D(*P.T, **kwargs)
	elif isinstance(P, Iterable):
		import numbers
		assert layout is not None, "missing layout"
		if angles is None:
			angles = np.repeat(60, len(P))
		elif isinstance(angles, numbers.Integral):
			angles = np.linspace(0, 2*np.pi, len(P), endpoint=False)
		assert len(angles) == np.prod(layout)
		if "fig" in kwargs.keys() and "ax" in kwargs.keys():
			fig, ax = kwargs["fig"], kwargs["ax"]
			kwargs.pop('fig', None)
			kwargs.pop('ax', None)
		else:
			fig, ax = plt.subplots(*layout, figsize=figsize)
		for i, p in enumerate(P):
			ax = fig.add_subplot(layout[0], layout[1], i+1, projection='3d')
			ax.scatter3D(*p.T, **kwargs) 
			ax.view_init(30, angles[i])
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
	return(fig, ax)

def rotating_disk(n_pixels: int, r: float, sigma: float = 1.0):
	from scipy.ndimage import gaussian_filter
	import numpy as np
	I = np.zeros(shape=(n_pixels, n_pixels))
	p = np.linspace(0, 1, n_pixels, False) + 1/(2*n_pixels) # center locations of pixels, in normalized space
	z = np.array([r, 0.0]).reshape((2,1))
	d = np.array([0.5, 0.5]).reshape((2,1))
	x,y = np.meshgrid(p, p)
	def disk_image(theta: float):
		R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		c = (R @ z) + d # center of disk in [0,1]^2
		D = np.flipud(np.sqrt((x - c[0])**2 + (y - c[1])**2))
		D[D <= r] = -1.0
		D[D > r] = 0.0
		D[D == -1.0] = 1.0
		return(np.ravel(gaussian_filter(D, sigma=1.0)).flatten())
	return(disk_image, 1.0)

# D = np.zeros(np.prod(x.shape))
# for i, (xi,yi) in enumerate(zip(x.flatten(),y.flatten())):
# 	p = np.array([xi,yi])
# 	D[i] = np.dot(p-b, u)# np.linalg.norm(z-v)

def white_bars(n_pixels: int, r: float, sigma: float = 1.0):
	''' 
	Returns a parameterization that yields a white vertical bar at various orientations in an image. 

	Fixed parameters: 
		n_pixels := number of pixels to make square image
		r := constant between [0,1] indicating how wide to make the bar 
		sigma := kernel parameter for gaussian blur
	Returns:
		bar := closure w/ parameters y_offset in [0, 1] and theta in [0, pi]
		c := normalizing constant for plotting
	'''
	from scipy.ndimage import gaussian_filter
	import numpy as np
	w = r
	p = np.linspace(0, 1, n_pixels, False) + 1/(2*n_pixels) # center locations of pixels, in normalized space
	x,y = np.meshgrid(p,p)
	c = np.array([0.5, 0.5]) 	# center of image
	def bar(theta: float, d: float):
		assert np.all(np.bitwise_and(d >= -1.0, d <= 1.0)), "d must be in the range [-1, 1]"
		assert np.all(np.bitwise_and(theta >= 0.0, theta <= np.pi)), "theta must be in the range [0, pi]"
		u = np.array([ 1.0, np.tan(theta) ])
		u = u / np.linalg.norm(u)
		c = np.array([0.5, 0.5]) 	# center of image
		d = d * (np.sqrt(2) / 2)  # scale where to place center of bar 
		if theta > np.pi/2:
			d = -d
		b = c + d*u 							# center of bar 
		D = np.zeros(np.prod(x.shape))
		for i, (xi,yi) in enumerate(zip(x.flatten(),y.flatten())):
			p = np.array([xi,yi])
			D[i] = np.dot(p-b, u)# np.linalg.norm(z-v)
		I = abs(D.reshape((n_pixels, n_pixels))).T
		I[I > w] = 1
		I = 1.0 - I
		return(gaussian_filter(I, sigma=sigma))
	c = np.max(bar(0.0, 0.0))
	return(bar, c)

	# u = np.array([ 1.0, np.tan(theta) ])
	# u = u / np.linalg.norm(u)
	# d = np.array([-di if ti <= np.pi/2 else di for di,ti in zip(d, theta)])*(np.sqrt(2) / 2)
	# U = np.c_[np.repeat(1.0, len(theta)), theta]
	# U = U / np.linalg.norm(U, axis = 1, keepdims = True)
	# B = c + d.reshape((len(d), 1)) * U 	# center of bars
	# D = [abs((x - b[0])*u[0] + (y - b[1])*u[1]).T for (u, b) in zip(U, B)]
	# # b = c + d*u 							# center of bar 
	# # D = (x - b[0])*u[0] + (y - b[1])*u[1]
	# # I = abs(D.reshape((n_pixels, n_pixels))).T
	# images = np.zeros((B.shape[0], n_pixels**2))
	# for i, img in enumerate(D):
	# 	img[img > w] = 1
	# 	img = 1.0 - img
	# 	images[i,:] = np.ravel(gaussian_filter(img, sigma=sigma).flatten())
	# return(images)

	# from scipy.ndimage import gaussian_filter
	# import numpy as np
	# w = r*np.sqrt(2)
	# p = np.linspace(0, 1, n_pixels, False) + 1/(2*n_pixels) # center locations of pixels, in normalized space
	# x,y = np.meshgrid(p,p)
	# def bar(y_offset: float, theta: float):
	# 	assert y_offset >= 0.0 and y_offset <= 1.0 
	# 	assert theta >= 0.0 and theta <= np.pi
	# 	# z = np.array([0.5, y_offset]) # intercept
	# 	# dist_to_line = np.cos(theta)*(z[1] - y) - np.sin(theta)*(z[0]-x)
	# 	# dist_to_line = ((y - y_offset)/np.tan(theta))*np.sin(theta)
	# 	# Z = np.array([np.array([xi,yi]) for xi,yi in zip(x.flatten(),y.flatten())])
	# 	# fig,ax = scatter2D(Z, c="blue")
	# 	# fig,ax = scatter2D(np.array(P), c="red", fig=fig, ax=ax)
	# 	# fig,ax = scatter2D(np.array([0.5, 0.5]), c="green", fig=fig, ax=ax)
	# 	# fig,ax = scatter2D(np.c_[x.flatten(), x.flatten()*m + b], c="purple", fig=fig, ax=ax)
		
	# 	m, b = np.tan(theta), y_offset
	# 	#pt = np.array([1.0, m + b])
	# 	z1 = np.array([0.50, b])
	# 	z2 = np.array([1.0, 1.0*m + b])
	# 	pt = z2 - z1
	# 	d = []
	# 	P = []
	# 	for xi,yi in zip(x.flatten(),y.flatten()):
	# 		u = pt / np.linalg.norm(pt)
	# 		v = np.array([xi,yi])
	# 		z = u*np.dot(v-np.array([0.5, b]), u)+np.array([0.5, b])
	# 		d.append(np.linalg.norm(z-v))
	# 		P.append(z)
	# 	dist_to_line = np.array(d)
	# 	# fig, ax = scatter2D(np.array(P))

	# 	I = abs(dist_to_line.reshape((n_pixels, n_pixels)))
	# 	I = np.flipud(I) # make origin lower-left, not top-left 
	# 	# I = (np.sqrt(2)/2)*(I/np.max(I))
	# 	I[I > w] = np.sqrt(2)
	# 	I = np.sqrt(2) - I ## invert intensity 
	# 	# I[I < (np.sqrt(2) - w)] = 0.0
	# 	# B = I.copy()
	# 	# I[I <= w] = -1.0
	# 	# I[I > w] = 0.0
	# 	# I[I == -1.0] = np.max(B[B <= w]) - B[B <= w] # 1.0 
	# 	return(gaussian_filter(I, sigma=sigma))
	# c = np.max(bar(0.0, 0.0))
	# return(bar, c)

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

	Parameters: 
		n_pixels := number of pixels wide/tall to make the resulting images 
		r := relative radius of dot (in (0, 1])
		n := (optional) number of samples desired 
		method := (optional) how to generate samples in the parameter space. Can be either "grid" or "random".
		mu := (optional) locations of dot centers to generate the dots at
	
	Returns: 
		samples := generated image samples 
		params	:= (x,y,i) parameters associated with each sample,
		f 			:= closure for generating more samples. See gaussian_blob() for more details. 
		c 			:= normalizing constant. See gaussian_blob() for more details. 
	'''
	assert r > 0 and r <= 1.0, "r must be in the range 0 < r <= 1.0"
	assert isinstance(n, int) or isinstance(n, tuple), "n must be integer of tuple of integers"
	ask_package_install("autograd")
	import numpy as np
	import autograd.numpy as auto_np

	## First generate the closure to make the images
	blob, c = gaussian_blob(n_pixels, r)

	if not(mu is None):
		samples = np.vstack([blob(auto_np.array([x,y])) for x,y in mu])
		params = mu 
	elif method == "random":
		## Generate uniformly random locations (in domain)
		assert n is not None, "'n' must be supplied if 'mu' is not."
		n1, n2 = (n, n) if isinstance(n, int) else (n[0], n[1])
		
		samples, params = [], []
		X, Y = np.random.uniform(size=n1,low=-r,high=1+r), np.random.uniform(size=n1,low=-r,high=1+r)
		for x,y in zip(X, Y):
			samples.append(blob(auto_np.array([x,y])))
			params.append([x, y, 1.0])
		
		NP = blob(auto_np.array([0.5, 0.5]))
		for t in np.random.uniform(size=n2, low=0.0, high=1.0):
			samples.append(t*NP)
			params.append([0.5, 0.5, 1-t])
		
		## Vertically stack 
		samples, params = np.vstack(samples), np.vstack(params)

	elif method == "grid":
		assert n is not None, "'n' must be supplied if 'mu' is not."
		if isinstance(n, int):
			n1, n2 = (n, n) 
		else:
			n1, n2 = (n[0], n[1])
		ng = int(np.floor(np.sqrt(n1)))
		samples, params = [], []
		for x in np.linspace(0.0-r,1.0+r,ng):
			for y in np.linspace(0.0-r,1.0+r,ng):
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
	''' 
	Generates stratified samples on a Mobius band embedded in R^3 

	To get uniform samples, N = (n_polar*n_wide) uniformly spaced coordinates are generated initially 
	from the intrinsic space of M. These points are converted to their extrinsic (3D) coordinates and 
	are then triangulated using a Delaunay triangulation. Finally, using the Delaunay triangles as stratum, 
	a stratified sampling scheme is employed by sampling randomly from each triangle using its barycentric 
	coordinates. This stratification ensures the samples are both sufficiently random but sufficiently "uniformly 
	spaced" around the band. 
	
	Returns: 
		- M := (n x 3) matrix of embedding coordinates 
		- B := (n x 2) matrix of intrinsic coordinates
		
	In the intrinsic coordinates, B[:,0] is the width parameter and B[:,1] is the angular coordinate
	'''

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
	return(MB, Y)


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
	a = np.hstack((a, np.zeros((a.shape[0], D - d))))

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
