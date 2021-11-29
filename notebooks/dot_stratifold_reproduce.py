import numpy as np
from tallem import TALLEM
from tallem.datasets import *
from tallem.samplers import landmarks
from tallem.cover import T, LandmarkCover
from tallem.alignment import procrustes
from itertools import product

# %% (1) Generate large data set 
samples, params, blob, norm_constant = white_dot(n_pixels=17, r=0.35, n=(1200, 100), method="grid")


# %% (1.a optional) Generate sample images
## Show random samples 
ind = np.random.choice(range(samples.shape[0]), size=5*10, replace=False)
fig, ax = plot_images(samples[ind,:], shape=(17,17), max_val=c, layout=(5,10), figsize=(6,3))

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



def grid_search_best(f, params, score_f):
  best_params, best_score = [None]*len(params.keys()),  np.inf
  for p in product(*params.values()):
    args = dict(zip(params.keys(), p))
    score = score_f(f(args))
    if score < best_score:
      best_score, best_params = score, p
  return(dict(zip(params.keys(), best_params)))
    

## Mark mrocrustes error to isomap as "best" performing
procrustes_error = lambda FX: procrustes(FX, x_iso)["distance"]

LLE = lambda kwargs: lle(X, 3, **kwargs)
LLE_params = dict(n_neighbors=[5,8,10,15,20,25], reg=[1e-2,1e-3,1e-4])
LLE_best = grid_search_best(LLE, params=LLE_params, score_f=procrustes_error)

HLLE = lambda kwargs: hessian_lle(X, 3, **kwargs)
HLLE_params = dict(n_neighbors=[15,20,25], reg=[1e-2,1e-3,1e-4], hessian_tol=[1e-3,1e-4,1e-5])
HLLE_best = grid_search_best(HLLE, params=HLLE_params, score_f=procrustes_error)

LTSA = lambda kwargs: ltsa(X, 3, **kwargs)
LTSA_params =  dict(n_neighbors=[5,8,10,15,20,25], reg=[1e-2,1e-3,1e-4])
LTSA_best = grid_search_best(LTSA, params=LTSA_params, score_f=procrustes_error)

LE = lambda kwargs: laplacian_eigenmaps(X, 3, **kwargs)
LE_params = dict(n_neighbors=[5,8,10,15,20,25], eigen_solver=['arpack','lobpcg'])
LE_best = grid_search_best(LE, params=LE_params, score_f=procrustes_error)


x_lle = lle(X, 3, **LLE_best)
x_hle = hessian_lle(X, 3, **HLLE_best)
x_ltsa = ltsa(X, 3, **LTSA_best)
x_le = laplacian_eigenmaps(X, 3, **LE_best)

# %% 
results = {}
for ns, sc in product([10, 14, 18, 22, 26, 28], [1.10, 1.45, 1.80, 2.20, 2.80, 3.20]):
  cover = LandmarkCover(X, n_sets=ns, scale=sc) # 20, 1.3
  top = TALLEM(cover, "pca3", D=3, pou="identity").fit(X)
  results[(ns, sc)] = procrustes(top.embedding_, x_iso)["distance"]

cover = LandmarkCover(X, n_sets=22, scale=2.2)
top = TALLEM(cover, "iso3", D=3, pou="identity").fit(X)
x_tallem = top.embedding_


# %% (5) Create the grid plot 
import colorcet as cc
from matplotlib import cm
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from tallem.color import bin_color
from tallem.distance import dist 

## To help design the spec 
from matplotlib.gridspec import GridSpec
def format_axes(fig):
  for i, ax in enumerate(fig.axes):
    ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    ax.tick_params(labelbottom=False, labelleft=False)

fig = plt.figure(constrained_layout=False, figsize=ratio*5)
gs = GridSpec(6,30,figure=fig,wspace=0, hspace=0)
for i in range(5):
  for j in range(5):
    ax = fig.add_subplot(gs[i, (2*j):(2*j+2)])

## lower left pole images
ax = fig.add_subplot(gs[5, 0:2])
ax = fig.add_subplot(gs[5, 2:4])
ax = fig.add_subplot(gs[5, 4:6])
ax = fig.add_subplot(gs[5, 6:8])
ax = fig.add_subplot(gs[5, 8:10])

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

format_axes(fig)


# %% Make actual figure 
viridis = cm.get_cmap('viridis', 100)
col_pal = [rgb2hex(viridis(i)) for i in range(100)]

## Use distance to black image as color gradient
d = dist(np.zeros_like(X[0,:]), X)
pt_color = list(bin_color(d, col_pal, scaling="equalize").flatten())


# %% Manually choose an orientation
from scipy.spatial.transform import Rotation as Rot
EMB = [x_tallem, x_pca, x_iso, x_mmds, x_lle, x_hle, x_ltsa, x_le]

# for i in range(50):
#   emb = (Rot.random(1, random_state=i).as_matrix() @ EMB[0].T).T
#   emb = np.array([e.T.flatten() for e in emb])
#   scatter3D(emb, c=pt_color)
#   plt.pause(0.50)
#   print(i)

emb = (Rot.random(1, random_state=35).as_matrix() @ EMB[0].T).T
emb = np.array([e.T.flatten() for e in emb])
EMB[0] = emb


# %% Re-orient them to all align with tallem
for i, emb in zip(range(1, len(EMB)), EMB[1:]):
  EMB[i] = procrustes(emb, EMB[0], coords=True)

# %% Make the figure 
ratio = np.array([5.41667, 716/1885 * 5.41667])# height/width ratio 
fig = plt.figure(constrained_layout=False, figsize=ratio*5, dpi=300)
gs = GridSpec(6,30,figure=fig,wspace=0.025, hspace=0.025)

# Draw 5x5 grid of evenly spaced samples
gs_img_params = dict(cmap='gray', vmin=0, vmax=norm_constant, aspect='auto')
params = product(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
indices = product(range(5), range(5))
for (i,j), (x,y) in zip(indices, params):
  ax = fig.add_subplot(gs[i, (2*j):(2*j+2)])
  img = blob([y, x]).reshape((17,17))
  ax.imshow(img, **gs_img_params)

# Pole images
for j, p in zip(range(5), np.linspace(1.0, 0.0, 5)):
  ax = fig.add_subplot(gs[5, (2*j):(2*j+2)]).imshow((p*blob([0.5, 0.5])).reshape((17,17)), **gs_img_params)
for ax in fig.axes[:30]: ax.axis('off')

LB_ROW, UB_ROW = [0,0,0,0,3,3,3,3], [3,3,3,3,6,6,6,6]
LB_COL, UB_COL = [10,15,20,25,10,15,20,25], [15,20,25,30,15,20,25,30]
TITLES = ["TALLEM", "PCA", "ISOMAP", "MDS", "LLE", "HLLE", "LTSA", "Laplacian Eigenmaps"]

for emb, title, i, j, k, l in zip(EMB, TITLES, LB_ROW, UB_ROW, LB_COL, UB_COL):
  ax = fig.add_subplot(gs[i:j,k:l], projection='3d')
  ax.scatter3D(*emb.T, c=pt_color, s=22, edgecolor='gray',linewidth=0.30)
  ax.set_title(title, y=-0.10, fontsize=18)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_zticklabels([])

plt.show()

## %% 
fig.savefig("stratifold.tiff", dpi=300)

# ax.set_zlim(-1.01, 1.01)
# fig.colorbar(surf, shrink=0.5, aspect=10)




# %% Just the dot images 
gc = 11
fig = plt.figure(constrained_layout=False, figsize=np.array([gc, gc])*5, dpi=300)
gs = GridSpec(gc,gc,figure=fig,wspace=0.025, hspace=0.025)

# Draw 5x5 grid of evenly spaced samples
gs_img_params = dict(cmap='gray', vmin=0, vmax=norm_constant, aspect='auto')
params = product(np.linspace(0.0, 1.0, gc), np.linspace(0.0, 1.0, gc))
indices = product(range(gc), range(gc))
for (i,j), (x,y) in zip(indices, params):
  ax = fig.add_subplot(gs[i,j])
  img = blob([y, x]).reshape((17,17))
  ax.imshow(img, **gs_img_params)

for ax in fig.axes: ax.axis('off')
plt.show()


## %% Pole images
fig = plt.figure(constrained_layout=False, figsize=np.array([gc, 1.0])*5, dpi=300)
gs = GridSpec(1,gc,figure=fig,wspace=0.025, hspace=0.025)
for j, p in zip(range(gc), np.linspace(1.0, 0.0, gc)):
  ax = fig.add_subplot(gs[0, j]).imshow((p*blob([0.5, 0.5])).reshape((17,17)), **gs_img_params)
for ax in fig.axes: ax.axis('off')
plt.show()