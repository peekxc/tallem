import numpy as np
from tallem import TALLEM
from tallem.datasets import *
from tallem.samplers import landmarks
from tallem.cover import LandmarkCover

# %% (1) Generate large data set 
samples, params, blob, c = white_dot(n_pixels=17, r=0.35, n=(1200, 100), method="grid")


# %% (1.a optional) Generate sample images
## Show random samples 
ind = np.random.choice(range(samples.shape[0]), size=5*10, replace=False)
fig, ax = plot_images(samples[ind,:], shape=(17,17), max_val=c, layout=(5,10), figsize=(6,3))

## Show parameterization
## TODO: show 5x5 samples of 2d manifold + 5 x 1 view of 1-D manifold 
inc = int(1200/25)


# %% (2) Generate smaller data set 
Lind, _ = landmarks(samples, k = 400)
ind = np.random.choice(Lind, size=6*10, replace=False)
X = np.vstack((samples[Lind,:], samples[-100:,:]))

# %% (3) Generate other embeddings to compare against 
from tallem.dimred import pca, isomap, lle, laplacian_eigenmaps, mmds, hessian_lle, ltsa
x_pca = pca(X, 3)
x_iso = isomap(X, 3)
x_mmds = mmds(X, 3)
x_lle = lle(X, 3)
x_hle = hessian_lle(X, 3, n_neighbors=15)
x_ltsa = ltsa(X, 3)
x_le = laplacian_eigenmaps(X, 3)



# %% 
cover = LandmarkCover(X, n_sets=20, scale=1.30) # 20, 1.3
top = TALLEM()


# %% (5) Create the grid plot 
import colorcet as cc
from matplotlib import cm
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from tallem.color import bin_color
from tallem.distance import dist 

viridis = cm.get_cmap('viridis', 100)
col_pal = [rgb2hex(viridis(i)) for i in range(100)]

## Use distance to black image as color gradient
# 5.41667 inches 716/1885
d = dist(np.zeros_like(X[0,:]), X)
pt_color = list(bin_color(d, col_pal, scaling="equalize").flatten())

# fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3), constrained_layout=True, squeeze=False)
ratio = np.array([5.41667, 716/1885 * 5.41667])
fig = plt.figure(figsize=ratio*5) # height/width ratio 

nc = 6

## To help design the spec 
from matplotlib.gridspec import GridSpec
def format_axes(fig):
  for i, ax in enumerate(fig.axes):
    ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    ax.tick_params(labelbottom=False, labelleft=False)

fig = plt.figure(constrained_layout=False, figsize=ratio*5)
gs = GridSpec(6,30,figure=fig)
for i in range(5):
  for j in range(5):
    ax = fig.add_subplot(gs[i, (2*j):(2*j+2)])

## first row of embeddings
ax = fig.add_subplot(gs[0:3, 10:15])
ax = fig.add_subplot(gs[0:3, 15:20])
ax = fig.add_subplot(gs[0:3, 20:25])
ax = fig.add_subplot(gs[0:3, 25:30])

## second row of embeddings
ax = fig.add_subplot(gs[3:6, 10:15])
ax = fig.add_subplot(gs[3:6, 15:20])
ax = fig.add_subplot(gs[3:6, 20:25])
ax = fig.add_subplot(gs[3:6, 25:30])

## lower left pole images
ax = fig.add_subplot(gs[5, 0:2])
ax = fig.add_subplot(gs[5, 2:4])
ax = fig.add_subplot(gs[5, 4:6])
ax = fig.add_subplot(gs[5, 6:8])
ax = fig.add_subplot(gs[5, 8:10])

format_axes(fig)


fig, axs = plt.subplots(*layout, figsize=figsize)
axs = axs.flatten()
for i, (img, ax) in enumerate(zip(P, axs)):
  #fig.add_subplot(layout[0], layout[1], i+1)
  plt.axis("off")
  ax.imshow(P[i,:].reshape(shape), cmap='gray', vmin=0, vmax=max_val, aspect='auto')
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

ax = fig.add_subplot(1, nc, 1, projection='3d')
ax.scatter3D(*x_pca.T, c=pt_color)
ax.set_title("PCA", y=-0.01)

ax = fig.add_subplot(2, nc, 3, projection='3d')
ax.scatter3D(*x_pca.T, c=pt_color)
ax.set_title("PCA", y=-0.01)

ax = fig.add_subplot(2, nc, 4, projection='3d')
ax.scatter3D(*x_iso.T, c=pt_color)
ax.set_title("ISOMAP", y=-0.01)

ax = fig.add_subplot(2, nc, 5, projection='3d')
ax.scatter3D(*x_mmds.T, c=pt_color)
ax.set_title("MDS", y=-0.01)

ax = fig.add_subplot(2, nc, 6, projection='3d')
ax.scatter3D(*x_lle.T, c=pt_color)
ax.set_title("LLE", y=-0.01)

ax = fig.add_subplot(2, nc, 9, projection='3d')
ax.scatter3D(*x_hle.T, c=pt_color)
ax.set_title("HLLE", y=-0.01)

ax = fig.add_subplot(2, nc, 10, projection='3d')
ax.scatter3D(*x_ltsa.T, c=pt_color)
ax.set_title("LTSA", y=-0.01)

ax = fig.add_subplot(2, nc, 11, projection='3d')
ax.scatter3D(*x_le.T, c=pt_color)
ax.set_title("Laplacian Eigenmaps", y=-0.01)

# ax.set_zlim(-1.01, 1.01)
# fig.colorbar(surf, shrink=0.5, aspect=10)



# %%
