# %% klein bottle 
import numpy as np 
from dreimac import CircularCoords

x = np.array([-1,-1,-1,0,0,0,1,1,1])
y = np.array([1,0,-1,1,0,-1,1,0,-1])

n_data = 10000
n_land = 300

theta = np.random.uniform(0,np.pi, n_data)
alpha = np.random.uniform(0,2*np.pi, n_data)

data = np.array([
  np.cos(alpha[i])*(np.cos(theta[i])*x + np.sin(theta[i])*y) +
  np.sin(alpha[i])*(np.cos(theta[i])*x + np.sin(theta[i])*y)**2 
  for i in range(n_data)
])

cc = CircularCoords(data, distance_matrix = False, n_landmarks= n_land, prime = 13)
circ_coord = cc.get_coordinates(perc=0.9, cocycle_idx=[0])

n_covers = 5
cent_covers = np.exp(1j*np.linspace(0,2*np.pi, n_covers, endpoint=False))
r_cover = 2*np.sin(np.pi/n_covers)
cover_inds =[np.abs(np.exp(1j*circ_coord) - center)<r_cover for center in cent_covers]

# Preallocate PoU
local_n_covers = n_covers
J = n_covers*local_n_covers
pou = np.zeros((n_data, J))

scale = 1.20 # 
kb_cover = {}
cc = 0
for j, inds in enumerate(cover_inds):
  cc_fiber = CircularCoords(data[inds], distance_matrix = False, n_landmarks= 100, prime = 7)
  circ_coord_fiber = cc_fiber.get_coordinates(cocycle_idx=[0])
  cent_covers = np.exp(1j*np.linspace(0,2*np.pi, local_n_covers, endpoint=False))
  r_cover_lower = 2*np.sin(np.pi/local_n_covers)
  cc_fiber_ind = np.flatnonzero(inds)
  for k, center in enumerate(cent_covers):
    jk_center_diff = np.abs(np.exp(1j*circ_coord_fiber) - center) # relative 
    jk_cover_bool = jk_center_diff < r_cover_lower
    jk_cover_ind = cc_fiber_ind[jk_cover_bool]
    pou[jk_cover_ind, cc] = jk_center_diff[jk_cover_bool] # pou step 
    kb_cover[(j,k)] = jk_cover_ind
    cc += 1


# np.sum([len(subset) for subset in kb_cover.values()])
# np.sum(pou != 0)

# cover = CircleCover(polar_coordinate, n_sets=5, scale=1.50)
from scipy.sparse import csc_matrix
from tallem import TALLEM
top = TALLEM(kb_cover, "pca2", D=3, pou=csc_matrix(pou))
top.fit(X=data)

from tallem.datasets import *
scatter3D(top.embedding_, c=circ_coord)

polar_coordinate = circ_coord

# %%
