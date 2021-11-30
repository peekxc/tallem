# %% Imports 
from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import *
from tallem.distance import dist 
from tallem.samplers import landmarks
from tallem.datasets import *
import matplotlib.pyplot as plt

# %% Load frey faces 
import pickle
ff = pickle.load(open('/Users/mpiekenbrock/tallem/data/frey_faces.pickle', "rb")).T # 20 x 28

# %% Run TALLEM on frey faces
from tallem.dimred import rnn_graph, knn_graph, geodesic_dist
#G = rnn_graph(ff)
G = knn_graph(ff, k=20)
D = floyd_warshall(G, directed=False)
cover = LandmarkCover(D, n_sets=25, scale=1.15)
# [len(subset) for subset in cover.values()]

top = TALLEM(cover, local_map="pca2", D=2)
emb = top.fit_transform(X=ff)

# %% Plot the points + the faces as images using matplotlib
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = plt.gca()
ax.axis('off')
plt.scatter(*emb.T, alpha=0.70, s=8, edgecolor='gray',linewidth=0.30)
Lind, Lrad = landmarks(emb, k = 120)
img_width = 0.02*np.max(dist(emb)) # size of plotted images
for i in Lind: 
	bbox = np.array([emb[i,0] + img_width*np.array([-1.0, 1.0]), emb[i,1] + img_width*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(ff[i,:].reshape((28,20)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255)
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0])-img_width, right=np.max(emb[:,0])+img_width)
ax.set_ylim(bottom=np.min(emb[:,1])-img_width, top=np.max(emb[:,1])+img_width)

# plt.savefig("frey_faces.png", dpi=300, format="png", pad_inches=0.0, transparent=True)
plt.show()

# %% (Optional) Plot the points + the faces as images using matplotlib + datashader
# import datashader as ds
# from datashader.mpl_ext import dsshow, alpha_colormap
# from pandas import DataFrame # bleh why is this necessary datashader
# import datashader.transfer_functions as tf
# from functools import partial

# fig = plt.figure(figsize=(8, 8), dpi=300)
# ax = plt.gca()
# ax.axis('off')
# df = DataFrame(emb, columns = ['x','y'])
# shade_hook=partial(tf.dynspread, threshold=0.000001, how='over')
# dsshow(df, ds.Point('x', 'y'), norm='eq_hist', aspect='equal', vmax=10, ax=ax, vmin=1)
# plt.show()

# da = DSArtist(ax_r[0], df, 'x', 'y', ds.mean('z'), norm = mcolors.LogNorm())

# %% MNIST eights 
import pickle
import pathlib
import scipy.io as sio
mn = pickle.load(open('/Users/mpiekenbrock/tallem/data/mnist_eights.pickle', "rb")).T # 28 x 28
rotate = lambda x: np.fliplr(x.T)
mn = np.array([rotate(mn[:,:,i]).flatten() for i in range(mn.shape[2])])
# pickle.dump(mn, open('/Users/mpiekenbrock/tallem/data/mnist_eights.pickle', 'wb'))
# mn = np.array([mn[:,:,i].flatten() for i in range(mn.shape[2])])

from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import *
from tallem.distance import dist 

cover = LandmarkCover(mn, k=15)
top = TALLEM(cover, local_map="pca2", n_components=2)
emb = top.fit_transform(X=mn, B=mn)

s = 0.02*np.max(dist(emb))

import matplotlib.pyplot as plt
fig, ax = scatter2D(emb, alpha=0.20, s=10)


from tallem.samplers import landmarks
Lind, Lrad = landmarks(emb, k = 120)
for i in Lind: 
	bbox = np.array([emb[i,0] + s*np.array([-1.0, 1.0]), emb[i,1] + s*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(mn[i,:].reshape((28,28)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255)
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0]), right=np.max(emb[:,0]))
ax.set_ylim(bottom=np.min(emb[:,1]), top=np.max(emb[:,1]))




from tallem.dimred import pca
emb = pca(mn, 2)
Lind, Lrad = landmarks(emb, k = 120)

fig, ax = scatter2D(emb, alpha=0.20, s=10)
for i in Lind: 
	bbox = np.array([emb[i,0] + s*np.array([-1.0, 1.0]), emb[i,1] + s*np.array([-1.0, 1.0])]).flatten()
	face_im = ax.imshow(mn[i,:].reshape((28,28)), origin='upper', extent=bbox, cmap='gray', vmin=0, vmax=255, aspect ='auto')
	face_im.set_zorder(20)
ax.set_xlim(left=np.min(emb[:,0]), right=np.max(emb[:,0]))
ax.set_ylim(bottom=np.min(emb[:,1]), top=np.max(emb[:,1]))
