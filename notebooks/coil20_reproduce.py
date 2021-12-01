import numpy as np
import imageio
import glob
from tallem.circular_coordinates import CircularCoords

# %% Load data set
coil_path = "/Users/mpiekenbrock/tallem/data/coil-20-unproc/*.png"
coil_imgs = [imageio.imread(im_path) for im_path in glob.glob(coil_path)]
coil_imgs = np.array([img.flatten() for img in coil_imgs])

# %% Compute circular coordinates
## Use geodesic distance?
from tallem.dimred import geodesic_dist, rnn_graph, knn_graph
# D = geodesic_dist(rnn_graph(cat_imgs, p=0.10).A)
# D = geodesic_dist(np.ascontiguousarray(knn_graph(coil_imgs, k=20).A, dtype=np.float64))

# cc = CircularCoords(D, n_landmarks=40, distance_matrix=True, prime=7, maxdim=1)
cc = CircularCoords(coil_imgs, n_landmarks=coil_imgs.shape[0], distance_matrix=False, prime=7, maxdim=1)
B = cc.get_coordinates(perc = 0.70, do_weighted = True, cocycle_idx=[0, 1, 2, 3, 4, 5], pou = "linear")


# %% isomap on circular coordinates 
from tallem.dimred import isomap
Z = isomap(coil_imgs, d=2)
plt.scatter(*Z.T, c=B)


# %% Plot 1d plot of circular coordinates
import matplotlib.pyplot as plt
get_bbox = lambda x,y,w: np.array([x + w*np.array([-1.0, 1.0]), y + w*np.array([-1.0, 1.0])]).flatten()
fig = plt.figure(figsize=(15, 4))
ax = plt.gca()
ax.eventplot(B, linelengths=0.25)
ax.set_ylim(bottom=0.75, top=1.7)
ax.set_xlim(left=-0.25, right=2*np.pi + 0.25)
for i in range(coil_imgs.shape[0]):
  plt.imshow(coil_imgs[i,:].reshape((416, 448)), extent=get_bbox(B[i], 1.45, 0.08), origin='upper', cmap='gray', vmin=0, vmax=255)
plt.show()


# %% Clustering
from tallem.distance import dist
from scipy.cluster.hierarchy import single, dendrogram, fcluster
Z = single(dist(coil_imgs))
# fig = plt.figure(figsize=(25, 10))
# dn = dendrogram(Z)
cl_assignments = fcluster(Z, t = 10000, criterion='distance')

# %% 
import matplotlib.pyplot as plt
get_bbox = lambda x,y,w: np.array([x + w*np.array([-1.0, 1.0]), y + 10/3*w*np.array([-1.0, 1.0])]).flatten()
fig = plt.figure(figsize=(10, 3), dpi=200)
ax = plt.gca()
ax.set_aspect('auto')
ax.eventplot(B, linelengths=0.25, lineoffsets=0.0)
ax.set_ylim(bottom=-0.20, top=5.7)
ax.set_xlim(left=-0.25, right=2*np.pi + 0.25)
for class_idx in range(1, 6):
  ind = np.flatnonzero(cl_assignments == class_idx)
  for i in ind:
    plt.imshow(coil_imgs[i,:].reshape((416, 448)), extent=get_bbox(B[i], class_idx, w=0.12), origin='upper', cmap='gray', vmin=0, vmax=255, aspect='auto')
plt.show()
# %%
