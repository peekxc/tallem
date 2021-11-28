
import numpy as np
import matplotlib.pyplot as plt
from tallem import TALLEM
from tallem.cover import CircleCover

np.random.seed(1234)
p = np.linspace(0, 2*np.pi, 100)
X = np.c_[np.cos(p), np.sin(p)]
# plt.scatter(*X.T)


cover = CircleCover(p, n_sets=4, scale=1.5, lb=np.min(p), ub=np.max(p)) # 20, 1.5 is best
pt_col = np.repeat("black_", X.shape[0])
set_colors = ["red", "blue", "yellow", "green"]
for i, ind in enumerate(cover.values()):
  for j in ind:
    pt_col[j] = "gray" if pt_col[j] != "black_" else set_colors[i]

## VERIFY 2D CASE WORKS
top = TALLEM(cover, local_map="pca2", D=2, pou="identity").fit(X=X)
plt.scatter(*top.embedding_.T, c=pt_col)

top = TALLEM(cover, local_map="iso2", D=2, pou="identity").fit(X=X)
plt.scatter(*top.embedding_.T, c=pt_col)

## Debug 1D case
top = TALLEM(cover, local_map="pca1", D=2, pou="identity").fit(X=X)
plt.scatter(*top.embedding_.T, c=pt_col)



plt.scatter(*X.T, c=pt_col)

import matplotlib.pyplot as plt 
i, j = 1, 2
ind_ij, idx_i, idx_j = np.intersect1d(top.cover[i], top.cover[j], return_indices = True)

## Verify procrustes
assert top.models[i].shape[1] == 1
plt.clf()
fi, fj = top.models[i][idx_i].flatten(), top.models[j][idx_j].flatten()
plt.eventplot(fi, colors=set_colors[i])
for fv, idx in zip(fi, ind_ij): 
  plt.text(x=fv, y=1.5, s=str(idx), size="x-small")
plt.eventplot(fj, colors=set_colors[j])
for fv, idx in zip(fj, ind_ij): 
  plt.text(x=fv, y=0.45, s=str(idx), size="x-small")
fig = plt.gcf()
fig.set_size_inches(12.5, 10.5)


# opa(fj[:,np.newaxis], fi[:,np.newaxis], scale=False)
# {
#  'rotation': array([[1.]]),
#  'scaling': 1,
#  'translation': array([2.22044605e-16]),
#  'distance': 3.961243160105721
# }
# opa(fi[:,np.newaxis], fj[:,np.newaxis], scale=False)
# {
#  'rotation': array([[1.]]),
#  'scaling': 1,
#  'translation': array([2.22044605e-16]),
#  'distance': 3.961243160105721
# }

from tallem.alignment import global_translations
TAUS = global_translations(cover, top.alignments)

## Get optimal translation vectors 
index_pairs = list(top.alignments.keys())
d = len(top.alignments[index_pairs[0]]['translation'])
rotations = { k: v['rotation'] for k,v in top.alignments.items() }
translations = { k: v['translation'] for k,v in top.alignments.items() }

## Evaluates pseudo-inverse on the coboundary matrix    
from tallem.sc import delta0D
J = len(cover)
S = (np.fromiter(range(J), dtype=int), index_pairs) ## vertex/edge simplicial complex
coboundary = delta0D(S, rotations)
coboundary[np.isnan(coboundary)] = 0.0
deltaX = np.linalg.pinv(coboundary)
shiftE = np.zeros(d*len(S[1]))
for (index,(i1,i2)) in enumerate(S[1]):
  shiftE[index*d:(index+1)*d] = translations[(i1,i2)]
shiftV = np.matmul(deltaX,shiftE)




print(top.alignments[(i,j)])
v_01 = top.alignments[(0,1)]['translation'] # vector translating u_1 |-> u_0 

# v_01 != f0.mean() - 1.0*f1.mean()

from scipy.linalg import orthogonal_procrustes
from tallem.alignment import opa, procrustes
from procrustes import orthogonal, rotational

FI, FJ = fi[:,np.newaxis], fj[:,np.newaxis]

opa(FJ, FI, scale=False)['rotation']
procrustes(FJ, FI, scale=False)['rotation']
rotational(FJ, FI, translate=True, scale=False)['t']
orthogonal(FJ, FI, translate=True, scale=False)['t']
orthogonal_procrustes(FJ-FJ.mean(axis=0), FI-FI.mean(axis=0))[0]


# %% Reflection testing
theta = np.pi/2
X = np.random.uniform(size=(15,2))
Re = np.array([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
Y = (Re @ X.T).T + np.array([-1.5, 0.75])

A = procrustes(X, Y, scale=False)
Z = procrustes(X, Y, scale=False, coords=True)
# Z4 = (A['scaling']*(A['rotation'] @ X.T).T + A['translation'])
plt.scatter(*X.T, c="blue")
plt.scatter(*Y.T, c="red")
plt.scatter(*Z.T, c="orange", alpha = 0.70)
# plt.scatter(*Z4.T, c="purple", alpha = 0.70)





R = procrustes(X.T, Y.T, scale=False)['rotation']


# %% Karcher mean idea
from geomstats.geometry.stiefel import Stiefel

# opa(f0[:,np.newaxis], f1[:,np.newaxis], scale=False)

## Get average frame phi_i averaged over all x
def initial_phi(top):
  d, D = top.d, top.D
  V = Stiefel(D, d)
  Phi = [None]*len(top.cover)
  for j in range(len(Phi)):
    point_ind = top.cover[j]
    phi_j = np.zeros((D, d))
    for i in point_ind:
      phi_hat = top._stf.generate_frame(j, np.array(top.pou[i,:].A.flatten(), dtype=np.float64))
      U, _, Vt = np.linalg.svd(top.A @ top.A.T @ phi_hat, full_matrices=False)
      phi_j += top.A.T @ (U @ Vt)
    phi_j /= len(point_ind)
    phi_j = V.projection(phi_j)
    Phi[j] = phi_j
  return(Phi)


## Precompute adjacency information
def neighbor_adjacency(top):
  neighbors = [np.array([], dtype=int)]*len(top.cover)
  E = list(top.alignments.keys())
  for (u,v) in E:
    neighbors[u] = np.append(neighbors[u], v)
    neighbors[v] = np.append(neighbors[v], u)
  neighbors = [np.sort(N) for N in neighbors]
  return(neighbors)

# np.linalg.norm(Phi[0]@top.alignments[(0,1)]['rotation'] - Phi[1])

## Get initial basepoints for Phi_j
Phi = initial_phi(top)
neighbors = neighbor_adjacency(top)
basepoints = [np.zeros((D, d)) for j in range(len(top.cover))]
for j in range(len(top.cover)):
  for k in neighbors[j]:
    if j < k:
      omega_ik = top.alignments[(j,k)]['rotation']
    else: 
      omega_ik = top.alignments[(k,j)]['rotation'].T
    basepoints[j] += Phi[j] @ omega_ik
  basepoints[j] /= len(neighbors[j])
  basepoints[j] = V.projection(basepoints[j])

## Compute Riemannian barycenter gradients
def phi_barycenter(top, Phi, step_size=0.01, max_iter=30):
  J, neighbors = len(top.cover), neighbor_adjacency(top)
  d, D = top.d, top.D
  V = Stiefel(D, d)
  assert len(Phi) == J and len(neighbors) == J
  D, d = top.D, top.d
  bc_gradients = [np.zeros((D, d)) for j in range(J)]
  next_basepoints = [np.zeros((D, d)) for j in range(J)]
  for j in range(len(top.cover)):
    ## Compute sum of gradients
    base_point = Phi[j]
    for k in neighbors[j]:
      omega_kj = top.alignments[(k, j)]['rotation'] if k < j else top.alignments[(j, k)]['rotation'].T
      bc_gradients[j] += V.canonical_metric.log(point=Phi[k] @ omega_kj, base_point=base_point)

    ## Update Phi's using (negative) gradient
    next_basepoints[j] = V.canonical_metric.exp(tangent_vec=-step_size*bc_gradients[j], base_point=base_point)
  print(np.sum([np.linalg.norm(g) for g in bc_gradients]))
  return(next_basepoints)



# %% Test 2D case for the TAU's: 
np.random.seed(1234)
p = np.linspace(0, 2*np.pi, 100)
X = np.c_[np.cos(p), np.sin(p)]+np.ones((100,2))*5
cover = CircleCover(p, n_sets=4, scale=1.5, lb=np.min(p), ub=np.max(p)) # 20, 1.5 is best
top = TALLEM(cover, local_map="pca2", D=2, pou="identity").fit(X=X)

## Initial (D x d)-dimensional Phi maps
Phi = initial_phi(top)

## Initial (D x 1)-dimensional tau's
taus_d = np.array(list(top.translations.values()))
Tau = np.array([Phi[j] @ taus_d[j,:] for j in range(len(top.cover))])

## Fixed d-dimensional translations from procrustes
V_trans = { k : a['translation'] for (k, a) in top.alignments.items() }

## Function to iteratively adjust tau's 
def update_taus(top, T, step_size):
  neighbors = neighbor_adjacency(top)
  for i, N in enumerate(neighbors):
    tau_i = np.zeros((top.D,))
    for k in N:
      if i < k:
        tau_i += Phi[i] @ V_trans[(i,k)]
      else: 
        tau_i += Phi[i] @ (-V_trans[(k,i)])
    tau_i /= len(N)
    T[i,:] = T[i,:] + step_size*tau_i
  return(T)

step_size = 0.001
new_Phi = phi_barycenter(top, Phi, step_size=step_size)
new_Tau = update_taus(top, Tau, step_size=step_size)

print(np.linalg.norm(Tau - new_Tau))
print(np.linalg.norm(np.array(new_Phi) - np.array(Phi)))

# tau1 = Taus_ext[0,:] + np.dot(Phi[0], V_trans[(0,1)])
# tau2 = Taus_ext[0,:] + np.dot(Phi[0], V_trans[(0,3)])

# Taus_ext[0,:] + np.dot(Phi[0], V_trans[0,:])
# Taus_ext[0,:] + np.dot(Phi[0], V_trans[0,:])

# %% Adjusted assembly 

## Custom assembly 
def custom_assembly(top, Phi, Tau):
  Z = np.zeros((top.n, top.D))
  for i in range(top.n):
    vp = top.pou[i,:].A.flatten()
    pou_ind = np.flatnonzero(vp)
    for j in pou_ind: 
      lm_point_ind = np.searchsorted(top.cover[j], i)
      f_xi = top.models[j][lm_point_ind,:]
      Z[i,:] += vp[j]*(np.dot(Phi[j],f_xi)+Tau[j,:])
  return(Z)

import matplotlib.pyplot as plt
step_size = 0.001
it = 0
while (it < 30):
  Z = custom_assembly(top, Phi, Tau)
  plt.scatter(*Z.T)
  plt.title(f"Iteration: {it}")
  plt.gca().set_aspect('equal')
  plt.pause(0.50)
  Phi = phi_barycenter(top, Phi, step_size=step_size)
  Tau = update_taus(top, Tau, step_size=step_size)
  print(Tau)
  it += 1


# %%
