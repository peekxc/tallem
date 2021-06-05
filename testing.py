# %% Need for interactive scripts 
import os 
os.chdir("/Users/mpiekenbrock/tallem")
%reload_ext autoreload
%autoreload 2

# %% Imports 
import numpy as np
from tallem.distance import dist
from tallem.isomap import isomap
from tallem.mds import classical_MDS

# %% Setup data 
X = np.random.uniform(size=(15,2))
Y = isomap(X, d = 2, k = 3)
x_mds = classical_MDS(dist(X, as_matrix=True))

# %% Plot local euclidean models
import matplotlib.pyplot as pyplot
pyplot.scatter(X[:,0], X[:,1])
pyplot.scatter(Y[:,0], Y[:,1])
pyplot.scatter(x_mds[:,0], x_mds[:,1])

# %% Verify MDS
from sklearn.manifold import MDS
metric_mds = MDS(n_components = 2, eps = 1e-14)
Y_fit = metric_mds.fit(X)
Y = Y_fit.embedding_
print(Y_fit.stress_)

pyplot.draw()
pyplot.scatter(X[:,0], X[:,1])
pyplot.scatter(Y[:,0], Y[:,1])
pyplot.show()

# %% Procrustes 
from tallem.procrustes import ord_procrustes
aligned = ord_procrustes(X, Y, transform=True)
Z = aligned["coordinates"]

pyplot.draw()
pyplot.scatter(X[:,0], X[:,1])
pyplot.scatter(Z[:,0], Z[:,1])
pyplot.show()

# %% SciPy Procrustes 
from scipy.spatial import procrustes as pro 
A, B, d = pro(X, Y)
pyplot.draw()
pyplot.scatter(A[:,0], A[:,1])
pyplot.scatter(B[:,0], B[:,1])
pyplot.show()

# %% SciPy Procrustes rotation 
from scipy.linalg import orthogonal_procrustes as orthog_pro
R, s = orthog_pro(X, Y)

# %% Test scaled Procrustes rotation 
R, s = orthog_pro(X, Y*0.50)

# %% Cover 
from tallem.landmark import landmarks
X = np.random.uniform(size=(30,2))
L = landmarks(X, k = 5)

# %% Draw the cover
pyplot.draw()
r = np.min(L['radii'])
fig, ax = pyplot.subplots()
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[L["indices"],0], X[L["indices"],1], s=10)
for lm in L['indices']:
	c = pyplot.Circle((X[lm,0], X[lm,1]), r, color='orange', alpha = 0.20)
	ax.add_patch(c)
pyplot.show()

# %% Partition of unity 
from tallem.isomap import partition_of_unity
P = partition_of_unity(X, centers = X[L['indices'],:], radius = r)

# %%
