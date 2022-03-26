
# %% Show grid of samples
import matplotlib
import numpy as np
from tallem.datasets import *

bar, c = white_bars(n_pixels=19, r=0.38, sigma=2.55)
samples = []
for d in np.linspace(-0.5, 0.5, num=9, endpoint=True):
	for theta in np.linspace(0, np.pi, num=11, endpoint=True):
		samples.append(np.ravel(bar(theta, d)).flatten())
samples = np.vstack(samples)
fig, ax = plot_images(samples, shape=(19,19), max_val=c, layout=(9,11))

# %% Oversample  + landmarks to get a uniform sampling
# n_params = 50
# X, B = np.zeros((n_params**2, 25**2)), np.zeros((n_params**2,2))
# cc = 0
# for d in np.linspace(-0.5, 0.5, num=n_params, endpoint=True):
# 	for theta in np.linspace(0, np.pi, num=n_params, endpoint=True):
# 		X[cc,:] = np.ravel(bar(theta, d)).flatten()
# 		B[cc,:] = np.array([theta, d])
# 		cc += 1
D = np.random.uniform(size=8000, low=-0.30, high=0.30)
Theta = np.random.uniform(size=8000, low=0, high=np.pi)
X = np.array([np.ravel(bar(theta, d)).flatten() for (theta, d) in zip(Theta, D)])
B = np.c_[Theta, D]

## Choose landmarks using the intrinsic metric 
from tallem.samplers import landmarks
Lind, Lrad = landmarks(X, 1200)
XL = X[Lind,:]
BL = B[Lind,:]

# ripser 
import ripser 


# %% Tallem on true polar coordinate
from tallem import TALLEM
from tallem.cover import CircleCover
polar_coordinate = BL[:,0]
# polar_coordinate = BL
cover = CircleCover(polar_coordinate, n_sets=16, scale=2.00, lb=np.min(polar_coordinate), ub=np.max(polar_coordinate)) # 20, 1.5 is best
top = TALLEM(cover, local_map="iso2", D=3, pou="quadratic")

emb = top.fit_transform(X=XL)
fig, ax = scatter3D(emb, c=polar_coordinate)

top.plot_nerve(X=XL, layout="hausdorff")

scatter3D(top.models[1], c= BL[top.cover[1],0])

# %% 
from tallem import TALLEM
from tallem.cover import LandmarkCover
from tallem.dimred import geodesic_dist, rnn_graph

D = rnn_graph(XL, p = 0.15).A
D = geodesic_dist(np.ascontiguousarray(D, dtype=np.float64))
cover = LandmarkCover(D, n_sets=16, scale=1.50) 
top = TALLEM(cover, local_map="pca3", D=3, pou="gaussian")
emb = top.fit_transform(X=XL)
fig, ax = scatter3D(emb, c=BL[:,0])


# %% High dimension - Isomap 
Y = top.assemble_high()
# Z = isomap(Y, d=3)
Z = pca(Y, d=3)
fig, ax = scatter3D(Z, c=polar_coordinate)


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

D = 2*B[:,0] # ensures range is [-0.5, 0.5]
Phi = B[:,1]/2.0 # ensures range is [0, pi]

X = np.array([np.ravel(bar(phi, d).flatten()) for (d,phi) in zip(D, Phi)])

XL = X
BL = Phi


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
(pts,dim,pm_params,ftype,dtype) = Patches_Mobius(8000, aspect=5, BW=0.4)
# plot_image(pts[0,:].reshape((17,17)))

from tallem.samplers import landmarks
# Lind, Lrad = landmarks(pts, k=1200)
Lind, Lrad = landmarks(pm_params, k=1200)
XL, BL = pts[Lind,:], pm_params[Lind,[0]]

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






# %% 
# %matplotlib
# fig, ax = scatter3D(top.embedding_, c=BL)
emb = top.fit_transform(X=XL)
fig, ax = scatter3D(emb, c=polar_coordinate)

## 3D images 
# Create a dummy axes to place annotations to
ax2 = fig.add_subplot(111,frame_on=False) 
ax2.axis("off")
ax2.axis([0,1,0,1])

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox

class ImageAnnotations3D():
	def __init__(self, xyz, imgs, ax3d,ax2d):
		self.xyz = xyz
		self.imgs = imgs
		self.ax3d = ax3d
		self.ax2d = ax2d
		self.annot = []
		for s,im in zip(self.xyz, self.imgs):
			x,y = self.proj(s)
			self.annot.append(self.image(im,[x,y]))
		self.lim = self.ax3d.get_w_lims()
		self.rot = self.ax3d.get_proj()
		self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

		self.funcmap = {"button_press_event" : self.ax3d._button_press,
										"motion_notify_event" : self.ax3d._on_move,
										"button_release_event" : self.ax3d._button_release}

		self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
										for kind in self.funcmap.keys()]

	def cb(self, event):
		event.inaxes = self.ax3d
		self.funcmap[event.name](event)

	def proj(self, X):
		""" From a 3D point in axes ax1, 
				calculate position in 2D in ax2 """
		x,y,z = X
		x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
		tr = self.ax3d.transData.transform((x2, y2))
		return self.ax2d.transData.inverted().transform(tr)

	def image(self,arr,xy):
		""" Place an image (arr) as annotation at position xy """
		im = offsetbox.OffsetImage(arr, zoom=0.50, cmap="gray")
		im.image.axes = ax
		ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
												xycoords='data', boxcoords="offset points",
												pad=0.025, arrowprops=dict(arrowstyle="->"))
		self.ax2d.add_artist(ab)
		return ab

	def update(self,event):
		if np.any(self.ax3d.get_w_lims() != self.lim) or \
									np.any(self.ax3d.get_proj() != self.rot):
			self.lim = self.ax3d.get_w_lims()
			self.rot = self.ax3d.get_proj()
			for s,ab in zip(self.xyz, self.annot):
					ab.xy = self.proj(s)

from tallem.samplers import landmarks
ind, _ = landmarks(XL, 30)
imgs = [img.reshape((25,25)) for img in XL[ind,:]]
ia = ImageAnnotations3D(emb[ind,:],imgs,ax,ax2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# %% Spinning animations
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
n_frames = 30 # 5 seconds * 15 fps 

EMB = CMOB_PCA3_CIRCLE

# def rotate_grid(EMB: Iterable, layout):
layout = (5, 6) # nrows, ncols
plt.clf()
plt.rcParams["font.size"] = "16"
fig = plt.figure(figsize=(60, 50))
fig.tight_layout(pad=0.08, h_pad=0, w_pad=0)
plt.axis('off')
AXES = []
# for i, ((nsets, scale), emb) in enumerate(EMB.items()): 
# for i, (params, emb) in enumerate(EMB): 
for i, (params, emb) in enumerate(EMB.items()):
	ax = fig.add_subplot(layout[0], layout[1], i+1, projection='3d') # (nrows, ncols, index), index is 1-based 
	scatter3D(emb, c=polar_coordinate, fig=fig, ax=ax, s=20.50)
	plt.axis('off')
	# EMB.print(params)
	plt.title(f"nsets={params[0]}, scale={params[1]}", y=-0.01)
	# plt.title(f"scale={params}", y=-0.01)
	AXES.append(ax)
	# return(AXES)

AZIMUTH = np.linspace(0, 360, n_frames)
ELEVATION = np.append(np.linspace(0, 90, int(n_frames/2)), np.linspace(90, 0, int(n_frames/2)))

def update_camera(frame, *fargs):
	for ax in AXES: 
		ax.view_init(elev=ELEVATION[frame], azim=AZIMUTH[frame])
		ax.autoscale(enable=True, axis='x', tight=True)
		ax.autoscale(enable=True, axis='y', tight=True)
		ax.autoscale(enable=True, axis='z', tight=True)
		ax.set_facecolor('none')
	plt.subplots_adjust(wspace=0, hspace=0)

anim = animation.FuncAnimation(fig, func=update_camera, frames=n_frames, interval=1000/15, repeat=True)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 60
anim.save('CMOB_PCA3_CIRCLE.gif', writer='imagemagick', fps=30)

# %% 
from tallem import TALLEM
from tallem.cover import CircleCover
polar_coordinate = BL[:,0]

CMOB_PCA3_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(polar_coordinate, n_sets=nset, scale=scale, lb=np.min(polar_coordinate), ub=np.max(polar_coordinate)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca3", D=3, pou="triangular")
		CMOB_PCA3_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)

# %% CMOB ISO3
CMOB_ISO3_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(polar_coordinate, n_sets=nset, scale=scale, lb=np.min(polar_coordinate), ub=np.max(polar_coordinate)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="iso3", D=3, pou="triangular")
		CMOB_ISO3_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)
		print((nset, scale))

CMOB_PCA2_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(polar_coordinate, n_sets=nset, scale=scale, lb=np.min(polar_coordinate), ub=np.max(polar_coordinate)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca2", D=3, pou="triangular")
		CMOB_PCA2_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)
		print((nset, scale))


CMOB_PCA1_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(polar_coordinate, n_sets=nset, scale=scale, lb=np.min(polar_coordinate), ub=np.max(polar_coordinate)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca1", D=3, pou="triangular")
		CMOB_PCA1_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)
		print((nset, scale))


from tallem.cover import LandmarkCover
CMOB_ISO3_LANDMARK = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = LandmarkCover(XL, n_sets=nset, scale=scale) # 20, 1.5 is best
		top = TALLEM(cover, local_map="iso3", D=3, pou="triangular")
		CMOB_ISO3_LANDMARK[(nset, scale)] = top.fit_transform(X=XL)
		print((nset, scale))


from tallem.dimred import isomap
CMOB_ISO3 = {} # 5 columns, 6 rows
for scale in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]:
	emb = isomap(XL, d=3, p=1.0)
	CMOB_ISO3[scale] = emb
	print(scale)

# from tallem.dimred import pca
# CMOB_PCA3 = {} # 5 columns, 6 rows
# for scale in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]:
# 	CMOB_PCA3[scale] = pca(XL, d=3)
# 	print(scale)

# %% Joshes 

from tallem.dimred import isomap
JMOB_ISO3 = {} # 5 columns, 6 rows
for scale in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]:
	emb = isomap(XL, d=3, p=1.0)
	CMOB_ISO3[scale] = emb
	print(scale)

JMOB_PCA3_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(BL, n_sets=nset, scale=scale, lb=np.min(BL), ub=np.max(BL)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca3", D=3, pou="triangular")
		JMOB_PCA3_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)

JMOB_PCA2_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(BL, n_sets=nset, scale=scale, lb=np.min(BL), ub=np.max(BL)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca2", D=3, pou="triangular")
		JMOB_PCA2_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)

# intrinsic
JMOB_PCA3_CIRCLE_LINT = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(BL, n_sets=nset, scale=scale, lb=np.min(BL), ub=np.max(BL)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca3", D=3, pou="triangular")
		JMOB_PCA3_CIRCLE_LINT[(nset, scale)] = top.fit_transform(X=XL)


# %% UMOB
UMOB_PCA3_CIRCLE = {} # 5 columns, 6 rows
for nset in [5, 10, 15, 20, 25]:
	for scale in [1.2, 1.4, 1.6, 2.0, 2.2, 2.4]:
		cover = CircleCover(BL, n_sets=nset, scale=scale, lb=np.min(BL), ub=np.max(BL)) # 20, 1.5 is best
		top = TALLEM(cover, local_map="pca3", D=3, pou="triangular")
		UMOB_PCA3_CIRCLE[(nset, scale)] = top.fit_transform(X=XL)



# %% 
