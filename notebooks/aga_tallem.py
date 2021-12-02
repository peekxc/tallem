# %% 
import numpy as np
from tallem.datasets import mobius_band, scatter3D
M, B = mobius_band(n_polar=120, n_wide=9, scale_band=0.15)


fig, ax = scatter3D(M, c=B[:,1])
ax.set_box_aspect((np.ptp(M[:,0]), np.ptp(M[:,1]), np.ptp(M[:,2])))

# %% 
from tallem import TALLEM
from tallem.cover import CircleCover, LandmarkCover
polar_coordinate = B[:,1]
# cover = LandmarkCover(M, n_sets=20, scale=1.20)
cover = CircleCover(polar_coordinate, n_sets=16, scale=1.15)
top = TALLEM(cover, "pca2", D=2, pou="identity")
top.fit(X=M)

# fig, ax = scatter3D(top.embedding_, c=B[:,1])
# ax.set_box_aspect((np.ptp(M[:,0]), np.ptp(M[:,1]), np.ptp(M[:,2])))

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
  # print(np.maximum(d_jk, d_kj))
  
# fix root
T = nx.minimum_spanning_tree(G) 

# tree_align = {}
# for (j,k) in T.edges():
#   tree_align[(j,k)] = top.alignments[(j,k)]

# from tallem.alignment import global_translations
# tree_translations = global_translations(top.cover, tree_align)


def gamma_tau(top, node_path, verbose=0):
  Gamma = np.eye(top.d)
  for cc in range(len(node_path)-1):
    k,l = node_path[cc], node_path[cc+1]
    omega_k = top.alignments[(k, l)]['rotation']
    # print(omega_k)
    Gamma = Gamma @ omega_k
    if verbose > 0: print(f"({k}, {l})", end='')
  if verbose > 0: print("")
  return(Gamma)

# Gammas, Taus = [], np.zeros(shape=(len(top.cover), top.d))
# root_node = 0
# for j in range(len(top.cover)):
#   node_path_to_j = nx.dijkstra_path(T, root_node, j) # j0 -> j1 -> ... -> j
#   node_path_to_j0 = nx.dijkstra_path(T, j, root_node) # j -> j-1 -> ... -> j0
#   gamma_j = gamma_tau(top, node_path_to_j) # len(node_path)-1 omegas, inclusive!  
#   tau_j = Taus[j,:]
#   path_len = len(node_path_to_j0)
#   for cc in range(len(node_path_to_j)-1):
#     k,l = node_path_to_j0[cc], node_path_to_j0[cc+1]
#     v = top.alignments[(k,l)]['translation'] # should be correct
#     r_path = node_path_to_j0[:(path_len-cc-1)]
#     if len(r_path) == 1: 
#       omega = np.eye(top.d)# identity 
#       # print("identity")
#     else: 
#       # print(r_path)
#       omega = gamma_tau(top, r_path, verbose=0)
#     tau_j += -(omega @ v)
#   Gammas.append(gamma_j)
#   Taus[j,:] = tau_j
    # print((k,l)) # for v




for (j,k) in top.alignments.keys():
  omega_kj = top.alignments[(k,j)]['rotation']
  v_jk = top.alignments[(j,k)]['translation']
  v_kj = top.alignments[(k,j)]['translation']
  # print(np.linalg.norm(omega_kj @ v_jk + v_kj))



# nx.draw(T, pos=)
Gammas, Taus = [], np.zeros(shape=(len(top.cover), top.d))
root_node = 0
for j in range(len(top.cover)):
  node_path = nx.dijkstra_path(T, root_node, j)
  omega_path = []
  for cc in range(len(node_path)-1):
    k, l = node_path[cc], node_path[cc+1]
    omega_lk = top.alignments[(l,k)]['rotation']
    v_lk = top.alignments[(l,k)]['translation']
    # v_kl = top.alignments[(k,l)]['translation']
    omega_path.append(omega_lk)
    Taus[l,:] = omega_lk @ Taus[k,:] - v_lk
  Gammas.append(gamma_tau(top, node_path)) 
  # test_gamma = np.eye(top.d)
  # for omega in omega_path:
  #   test_gamma = test_gamma @ omega
  #   print(test_gamma)

  # Taus[l,:] = omega_lk @ Taus[k,:] - v_kl
  # print(Taus)
    # if k < l: 
    #   omega_kl = top.alignments[(k, l)]['rotation']
    #   v_kl = top.alignments[(k,l)]['translation']
    #   omega_path.append(omega_kl)
    #   Taus[l,:] = omega_kl.T @ (Taus[k,:] + v_kl)
    # else: 
    #   omega_lk = top.alignments[(l, k)]['rotation']
    #   v_lk = top.alignments[(l,k)]['translation']
    #   omega_path.append(omega_kl.T)
    #   Taus[j,:] = omega_lk @ (Taus[j,:] - v_lk)

  # Gamma = np.eye(top.d)
  # for omega in omega_path:
  #   Gamma = Gamma @ omega
  # Gammas.append(Gamma)

## Get Y 
# Y_ind = np.array(list(range(M.shape[0])))
# for (j,k) in T.edges:
#   jk_ind = np.intersect1d(top.cover[j], top.cover[k])
#   Y_ind = np.setdiff1d(Y_ind, jk_ind)

T_edges = list(T.edges())
Y_ind = np.array(list(range(M.shape[0])))
for (j,k) in G.edges:
  if not((j,k) in T_edges):
    jk_ind = np.intersect1d(top.cover[j], top.cover[k])
    Y_ind = np.setdiff1d(Y_ind, jk_ind)
    #print((j,k))

def aga_assembly(top, Phi, Tau, ind=None):
  if ind is None: ind = list(range(top.n))
  Z = np.zeros((len(ind), top.d))
  for i, idx in enumerate(ind):
    vp = top.pou[idx,:].A.flatten()
    pou_ind = np.flatnonzero(vp)
    for j in pou_ind: 
      lm_point_ind = np.searchsorted(top.cover[j], idx)
      f_xi = top.models[j][lm_point_ind,:]
      Z[i,:] += vp[j]*Phi[j] @ (f_xi+Tau[j,:])
  return(Z)

## add the tau's togther like the omegas 

#Taus = np.array(list(top.translations.values()))
# Taus = np.array(list(tree_translations.values())) 
Z = aga_assembly(top, Gammas, Taus, Y_ind)


import matplotlib.pyplot as plt
plt.scatter(*Z.T, c=polar_coordinate[Y_ind])
