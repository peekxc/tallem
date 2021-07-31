# %% Imports 
from src.tallem import TALLEM
from src.tallem.cover import IntervalCover
from src.tallem.datasets import mobius_band
from src.tallem.mds import classical_MDS
from src.tallem.distance import dist

# %% Setup parameters
X, B = mobius_band(n_polar=26, n_wide=6, embed=3).values()
B_polar = B[:,1].reshape((B.shape[0], 1))
cover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, gluing=[1])
f = lambda x: classical_MDS(dist(x, as_matrix=True), k = 2)

# %% Run TALLEM
%%time
embedding = TALLEM(cover=cover, local_map=f, n_components=3)
X_transformed = embedding.fit_transform(X, B_polar)

# %% Draw a 3D projection
import matplotlib.pyplot as plt
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], marker='o', c=B[:,0])











# TODO: add gradients to tox, nose, or pytest testing https://tox.readthedocs.io/en/latest/