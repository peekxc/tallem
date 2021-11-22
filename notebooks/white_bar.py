
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
# n_params = 50
# X, B = np.zeros((n_params**2, 25**2)), np.zeros((n_params**2,2))
# cc = 0
# for d in np.linspace(-0.5, 0.5, num=n_params, endpoint=True):
# 	for theta in np.linspace(0, np.pi, num=n_params, endpoint=True):
# 		X[cc,:] = np.ravel(bar(theta, d)).flatten()
# 		B[cc,:] = np.array([theta, d])
# 		cc += 1
D = np.random.uniform(size=8000, low=-0.5, high=0.50)
Theta = np.random.uniform(size=8000, low=0, high=np.pi)
X = np.array([np.ravel(bar(theta, d)).flatten() for (theta, d) in zip(Theta, D)])
B = np.c_[Theta, D]

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
cover = CircleCover(polar_coordinate, n_sets=15, scale=1.5, lb=np.min(polar_coordinate), ub=np.max(polar_coordinate)) # 20, 1.5 is best
top = TALLEM(cover, local_map="pca3", D=3, pou="triangular").fit(X=XL)
print(top)

# %% Nerve complex 
top.plot_nerve(X=XL, layout="hausdorff")

# import matplotlib.pyplot as plt
# plt.eventplot(polar_coordinate, orientation='horizontal')

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
J = int(np.sqrt(XL[ind,:].shape[1]))
ind = np.random.choice(top.cover[7], size=50)
fig, ax = plot_images(XL[ind,:], shape=(J,J), layout=(5,10), figsize=(8,4))

# %% 
ind = np.random.choice(top.cover[13], size=50)
fig, ax = plot_images(XL[ind,:], shape=(25,25), max_val=c, layout=(5,10), figsize=(8,4))


# %% 
from tallem import TALLEM
from tallem.cover import LandmarkCover
cover = LandmarkCover(XL, n_sets=15, scale=1.25)
top = TALLEM(cover, local_map="pca3", D=3, pou="triangular").fit(X=XL)
print(top)

# %% Nerve complex 
top.plot_nerve(X=XL, layout="hausdorff")

# %% 
fig, ax = scatter3D(top.embedding_, c=BL[:,0])


# %% Isomap 
from tallem.distance import dist
from tallem.dimred import isomap, connected_radius, enclosing_radius

## make many isomaps
pc, pe = connected_radius(dist(XL, as_matrix=True)), enclosing_radius(dist(XL, as_matrix=True))
all_isos = [isomap(XL, d=3, r=p) for p in np.linspace(pc, pe, 30)]

%matplotlib	
fig, ax = scatter3D(all_isos[29], c=BL[:,0])


# %% Parameterize using the mobius band directly
import numpy as np 
from tallem.dimred import isomap
from tallem.datasets import white_bars, mobius_band


M, B = mobius_band()
bar, c = white_bars(n_pixels=25, r=0.23, sigma=2.25)

D = 4*B[:,0] # ensures range is [-0.5, 0.5]
Phi = B[:,1]/2.0 # ensures range is [0, pi]

X = np.array([np.ravel(bar(phi, d).flatten()) for (d,phi) in zip(D, Phi)])
 
from tallem.datasets import * 
%matplotlib
fig, ax = scatter3D(isomap(X, d=3), c=Phi)


from tallem import TALLEM
from tallem.cover import LandmarkCover
cover = LandmarkCover(X, n_sets=15, scale=1.5)
top = TALLEM(cover, local_map="pca2", D=3, pou="triangular").fit(X=X)
print(top)

fig, ax = scatter3D(top.embedding_, c=Phi)


top.plot_nerve(X=X, layout="hausdorff")

from tallem.cover import CircleCover
cover = CircleCover(Phi, n_sets=15, scale=1.5, lb=0, ub=np.pi) # 20, 1.5 is best
top = TALLEM(cover, local_map="pca2", D=3, pou="triangular").fit(X=X)
fig, ax = scatter3D(top.embedding_, c=Phi)

# %% Joshes code

def Patches_Mobius(numpts,BW=1,aspect=3,ranseed = 123):
	dim = aspect**2
	
	np.random.seed(ranseed)
	pts = np.empty([numpts,dim])
	
	angles = [np.random.uniform(-np.pi/2,np.pi/2) for x in range(numpts)]
	offset = [np.random.uniform(-BW,BW) for x in range(numpts)]
	
	labels = np.transpose(np.array([angles,offset]))
	angles = np.array(angles)
	offset = np.array(offset)
	
	XYvals = [-1+2*x/(aspect-1) for x in range(aspect)]
	for i in range(dim):
		xval = XYvals[i%aspect]
		yval = XYvals[int(i/aspect)]
		vec1 = [xval,yval]
		for j in range(numpts):
			vec2 = [np.sin(angles[j]),np.cos(angles[j])]
			dist = np.dot(vec1,vec2) - offset[j]
			pts[j,i] = 1/(1+np.exp(dist**2))
	
	ftype = 'conts'
	dtype = 'euclidean'
	
	return (pts,dim,labels,ftype,dtype)


import numpy as np
(pts,dim,params,ftype,dtype) = Patches_Mobius(10000, aspect=17, BW=0.4)
# plot_image(pts[0,:].reshape((17,17)))

from tallem.samplers import landmarks
Lind, Lrad = landmarks(pts, k=1500)
XL, BL = pts[Lind,:], params[Lind,[0]]

# X = np.array([[np.cos(2*a),np.sin(2*a)] for a in params[:,0]])



from tallem.dimred import isomap, pca
fig, ax = scatter3D(isomap(XL, d=3), c=BL) 

fig, ax = scatter3D(pca(XL, d=3), c=BL) 


from tallem import TALLEM
from tallem.cover import CircleCover, LandmarkCover
cover = CircleCover(BL, n_sets=15, scale=1.0, lb=np.min(BL), ub=np.max(BL)) # 20, 1.5 is best
top = TALLEM(cover, local_map="pca3", D=3, pou="quadratic")
top = top.fit(X=XL)

top.plot_nerve(X=XL, layout="hausdorff")

%matplotlib
fig, ax = scatter3D(top.embedding_, c=BL)


from tallem.cover import LandmarkCover
cover = LandmarkCover(XL, n_sets=15, scale=1.5) # 20, 1.5 is best
top = TALLEM(cover, local_map="pca2", D=3, pou="identity")
top = top.fit(X=XL)
fig, ax = scatter3D(top.embedding_, c=BL)


# %% Begin interactive nerve plot code 


from ripser import ripser
from persim import plot_diagrams
dgm = ripser(XL, coeff=3)

plot_diagrams(dgm['dgms'], show=True)

