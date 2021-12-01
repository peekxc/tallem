# %% 
from tallem.datasets import mobius_band
M, B = mobius_band()

# %% 
from tallem import TALLEM
from tallem.cover import CircleCover, LandmarkCover
polar_coordinate = B[:,1]
# cover = LandmarkCover(M, n_sets=20, scale=1.20)
cover = CircleCover(polar_coordinate, n_sets = 8, scale=1.20)
top = TALLEM(cover, "pca2", D=3, pou="identity")
top.fit(X=M)

# %% 
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from itertools import combinations
from tallem.dimred import cmds

import networkx as nx
G = nx.Graph()
G.add_nodes_from(range(len(top.cover)))
for (j,k) in top.alignments.keys():
  d_jk = directed_hausdorff(M[cover[j],:], M[cover[k],:])[0]
  d_kj = directed_hausdorff(M[cover[k],:], M[cover[j],:])[0]
  G.add_edge(j, k, weight=np.maximum(d_jk, d_kj))
  
# fix root
T = nx.minimum_spanning_tree(G) 
# nx.draw(T, pos=)
Gammas = []
root_node = 0
for j in range(len(top.cover)):
  node_path = nx.dijkstra_path(T, root_node, j)
  omega_path = []
  for c, k in enumerate(node_path[:-1]):
    l = node_path[c+1]
    if k < l: 
      omega_path.append(top.alignments[(k, l)]['rotation'])
    else: 
      omega_path.append(top.alignments[(l, k)]['rotation'].T)
  Gamma = np.eye(top.d)
  for omega in omega_path:
    Gamma = Gamma @ omega
  Gammas.append(Gamma)

## Get Y 
Y_ind = np.array(list(range(M.shape[0])))
for (j,k) in top.alignments.keys():
  jk_ind = np.intersect1d(top.cover[j], top.cover[k])
  Y_ind = np.setdiff1d(Y_ind, jk_ind)


def aga_assembly(top, Phi, Tau, ind=None):
  if ind is None: ind = list(range(top.n))
  Z = np.zeros((len(ind), top.d))
  for i, idx in enumerate(ind):
    vp = top.pou[idx,:].A.flatten()
    pou_ind = np.flatnonzero(vp)
    for j in pou_ind: 
      lm_point_ind = np.searchsorted(top.cover[j], idx)
      f_xi = top.models[j][lm_point_ind,:]
      Z[i,:] += vp[j]*np.dot(Phi[j],f_xi+Tau[j,:])
  return(Z)

Taus = np.array(list(top.translations.values()))
Z = aga_assembly(top, Gammas, Taus, Y_ind)

import matplotlib.pyplot as plt
plt.scatter(*Z.T, c=polar_coordinate[Y_ind])


from tallem.datasets import *
aga_coords = np.c_[Z, polar_coordinate[Y_ind]]
scatter3D(aga_coords, c=polar_coordinate[Y_ind])
