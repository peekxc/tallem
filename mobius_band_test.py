# %% 
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

# %% Mobius band example
import numpy as np
from src.tallem import TALLEM
from src.tallem.cover import IntervalCover
from src.tallem.datasets import mobius_band

## Generate mobius band + polar coordinate 
M = mobius_band(n_polar=66, n_wide=9, scale_band = 0.25, plot=False, embed=6)
X, B = M['points'], M['parameters'][:,[1]]

## Assemble the embedding with TALLEM
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(B, n_sets = 10, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)
emb = TALLEM(cover, local_map="pca2", n_components=3).fit_transform(X, B)

## Visualize the resulting embedding
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*emb.T, c = B)