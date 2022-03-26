import matplotlib
import numpy as np 
import matplotlib.pyplot as plt
from tallem.dimred import pca

theta = np.linspace(0, 4*(2*np.pi), 20*4+1, endpoint=True)
y = np.c_[theta, np.cos(theta % (2*np.pi))*2]

fig = plt.figure(figsize=(10, 4), dpi=250)
ax = plt.gca()
ax.plot(*y.T, c="gray", linewidth=0.75)
ax.scatter(*y.T, zorder=10, s=14)
ax.set_aspect('equal')
ax.axis('off')

# ind = np.flatnonzero(abs(y[:,1]) < 10*np.finfo(np.float64).resolution)

# def proj_subspace(x, subspace, base_point = None):
#   if base_point is None: 
#     base_point


def neighborhood_cover(X, d, method=["tangent_proj"], ind = None, include_self=True, **kwargs):
  ''' 
  Constructs a weighted cover by constructing a neighborhood around every point and then 
  computing some predefined local functional on the points within each neighborhood

  Parameters: 
    X := (n x d) point cloud data in Euclidean space, or and (n x n) sparse adjacemency matrix yielding a weighted neighborhood graph
    d := local dimension where the metric is approximately Euclidean
    method := choice of neighborhood criteria

  Returns: 
    cover := (n x J) csc_matrix 
    weights := J-length array of weights computed using 'method'
  '''
  from tallem.dimred import neighborhood_graph, neighborhood_list, pca
  from scipy.sparse import csr_matrix, csc_matrix, find
  symmetrize = False 
  if "k" in list(kwargs.keys()):
    symmetrize = True 
  G = neighborhood_graph(X, **kwargs) if ind is None else neighborhood_list(X[ind,:], X, **kwargs)
  if include_self:
    for i in range(G.shape[0]):
      G[i,i] = True
  G = G.astype(bool)
  if symmetrize:
    G = csc_matrix(G.A + G.A.T)
  r,c,v = find(G)
  weights = np.zeros(X.shape[0])
  tangents = [None]*X.shape[0]
  for i, x in enumerate(X): 
    nn_idx = c[r == i] #np.append(np.flatnonzero(G[i,:].A), i)
    if len(nn_idx) < 2: 
      raise ValueError("bad cover")
    centered_pts = X[nn_idx,:]-x
    _, T_y = pca(centered_pts, d=d, coords=False)
    tangents[i] = T_y # ambient x local
    proj_coords = np.dot(centered_pts, T_y) # project points onto d-tangent plane
    proj_points = np.array([np.sum(p*T_y, axis=1) for p in proj_coords]) # orthogonal projection in D dimensions
    weights[i] = np.sum([np.sqrt(np.sum(diff**2)) for diff in (centered_pts - proj_points)]) # np.linalg.norm(centered_pts - proj_points)
  return(G, weights, tangents)

# (centered_pts @ T_y)[0,0]*T_y[:,0] + (centered_pts @ T_y)[0,1]*T_y[:,1]

# plt.scatter(*X.T)
# plt.scatter(*X[nn_idx,:].T, c="red")

n = y.shape[0]
n_neighbors, s = 3, 1.5
mean_proj = np.zeros(shape=(n,))
for i in range(y.shape[0]):
  nn_idx = np.array([x for x in range(i-n_neighbors, i+n_neighbors+1) if x >= 0 and x < n], dtype=int)
  _, T_y = pca(y[nn_idx,:]-y[i,:], d=1, coords=False)
  p = np.vstack([y[i,:] + s*T_y.T, y[i,:] - s*T_y.T])
  # plt.plot(*p.T, c="red", alpha=0.30)
  centered_pts = y[nn_idx,:]-y[i,:]
  proj_points = np.dot(centered_pts, T_y)*T_y.T
  mean_proj[i] = np.linalg.norm(centered_pts - proj_points)

from tallem.color import bin_color, linear_gradient
import colorcet as cc

fig = plt.figure(figsize=(10, 4), dpi=250)
ax = plt.gca()
for i in range(y.shape[0]):
  nn_idx = np.array([x for x in range(i-n_neighbors, i+n_neighbors+1) if x >= 0 and x < n], dtype=int)
  _, T_y = pca(y[nn_idx,:]-y[i,:], d=1, coords=False)
  for ni in nn_idx: 
    ax.plot([y[ni,0], y[i,0]], [y[ni,1], y[i,1]], c="black", linewidth=0.40)
  p = np.vstack([y[i,:] + s*T_y.T, y[i,:] - s*T_y.T])
  ax.plot(*p.T, c="red", alpha=0.30)
# ax.scatter(*y.T, c=mean_proj, s=65, edgecolors="gray", linewidths=0.50, zorder=10)
ax.scatter(*y.T, s=14, zorder=10)
ax.set_aspect('equal')
ax.axis('off')

from tallem.set_cover import greedy_weighted_set_cover

def create_cover(k=5):
  cover = np.zeros(shape=(y.shape[0], y.shape[0]))
  for i in range(y.shape[0]):
    nn_idx = np.array([x for x in range(i-k, i+k+1) if x >= 0 and x < n], dtype=int)
    cover[nn_idx, i] = 1.0
  return(cover)

from scipy.sparse import csc_matrix, data
cover = create_cover(5)
best_ind = greedy_weighted_set_cover(y.shape[0], csc_matrix(cover), mean_proj)

## Plot the tangent space vectors, colored by weights, and the set cover solution 
fig = plt.figure(figsize=(10, 4), dpi=250)
ax = plt.gca()
s = 2.45
for i in range(y.shape[0]):
  nn_idx = np.array([x for x in range(i-n_neighbors, i+n_neighbors+1) if x >= 0 and x < n], dtype=int)
  _, T_y = pca(y[nn_idx,:]-y[i,:], d=1, coords=False)
  p = np.vstack([y[i,:] + s*T_y.T, y[i,:] - s*T_y.T])
  if i in best_ind:
    ax.plot(*p.T, c="red", alpha=0.70)
ax.scatter(*y.T, c=mean_proj, s=25, edgecolors="gray", linewidths=0.50, zorder=10)
ax.scatter(*y[best_ind, :].T, c="red", s=55, edgecolors="gray", linewidths=0.50, alpha=1.0, zorder=20)
ax.set_aspect('equal')
ax.axis('off')

from scipy.sparse import csc_matrix
from scipy.optimize import linprog

## TODO: randomized approach


# %%  PySAT solution 
import pysat
from pysat.formula import WCNF

## Make the weighted formula 
wcnf = WCNF()
subset_weights = np.array(mean_proj*1000, dtype=int)
for j in range(cover.shape[1]): 
  # weight is set to None by default meaning that the clause is hard
  wcnf.append(list(np.flatnonzero(cover[:,j])+1), weight=None) 

for j, w in enumerate(subset_weights): 
  # prefer excluding the subset => find minimal weight solution 
  wcnf.append([-int(j+1)], weight=w)

wcnf.to_file("cosine_wcnf.cnf")
wcnf = WCNF(from_file='cosine_wcnf.cnf')
# wcnf.to_alien(open("cosine_wncf.smt", 'w'), format='smt')


def maxsat_wcnf(cover, weights):
  from pysat.formula import WCNF
  assert cover.shape[1] == len(weights)
  cover = cover.astype(bool) 
  wcnf = WCNF()
  subset_weights = np.array(weights, dtype=float)
  for j in range(cover.shape[1]): 
    wcnf.append(list(np.flatnonzero(cover[:,j].A)+1), weight=None)
  for j, w in enumerate(subset_weights): 
    wcnf.append([-int(j+1)], weight=w)
  wcnf.to_file("cosine_wcnf.cnf")
  wcnf = WCNF(from_file='cosine_wcnf.cnf')
  return(wcnf)

# RC2(wcnf)

## Solve 
from pysat.examples.rc2 import RC2
# with RC2(wcnf) as rc2: 
#   for assignment in rc2.enumerate():
#     print('model {0} has cost {1}'.format(assignment, rc2.cost))

# TODO: iterate through k-subsets!
for k in range(2, 8):
  cover = create_cover(k)
  formula = maxsat_wcnf(cover, mean_proj)
  assignment = RC2(formula).compute()
  sol_ind = np.flatnonzero(np.array(assignment) >= 0)
  print(f"size: {len(sol_ind)}, weight: {np.sum(mean_proj[sol_ind])}")

## Ensure solution is valid




sol_ind = np.flatnonzero(np.array(assignment) >= 0)
fig = plt.figure(figsize=(10, 4), dpi=300)
ax = plt.gca()
ax.scatter(*y.T, c=mean_proj, s=65, edgecolors="gray", linewidths=0.50, zorder=10)
ax.scatter(*y[sol_ind, :].T, c="red", s=85, edgecolors="gray", linewidths=0.50, alpha=1.0, zorder=20)


membership = np.zeros((cover.shape[0],), dtype=bool)
for j in sol_ind: 
  membership[np.flatnonzero(cover[:,j])] = True
assert np.all(membership)


# %% RC
def maxsat_wcnf(cover, weights):
  from pysat.formula import WCNF
  assert cover.shape[1] == len(weights)
  cover = cover.astype(bool) 
  wcnf = WCNF()
  subset_weights = np.array(weights, dtype=float)
  for j in range(cover.shape[1]): 
    wcnf.append(list(np.flatnonzero(cover[:,j].A)+1), weight=None)
  for j, w in enumerate(subset_weights): 
    wcnf.append([-int(j+1)], weight=w)
  wcnf.to_file("cosine_wcnf.cnf")
  wcnf = WCNF(from_file='cosine_wcnf.cnf')
  return(wcnf)

def weighted_set_cover_rc2(cover, weights, **kwargs):
  from pysat.examples.rc2 import RC2
  formula = maxsat_wcnf(cover, weights)
  assignment = RC2(formula, **kwargs).compute()
  sol_ind = np.flatnonzero(np.array(assignment) >= 0)
  return(sol_ind)


# %% Z3 solution
from z3 import *
from z3 import Or, Bool

def weighted_set_cover_z3(cover, weights, timeout=1000):
  from scipy.sparse import issparse
  assert cover.shape[1] == len(weights)
  assert issparse(cover)

  ## Initialize 
  n, J = cover.shape
  opt = Optimize()

  ## Add weights for each subset 
  weight_vars = [Real('w%f' % j) for j in range(J)]
  for j in range(J):
    opt.add(weight_vars[j] == weights[j])

  ## Add subset indicators
  subset_vars = [Int('s%d' % j) for j in range(J)]

  ## Add pointwise constraints
  for s in subset_vars: opt.add(Or(s == 0, s == 1))
  for i in range(n):
    ind = np.flatnonzero(cover[i,:].A)
    if (len(ind) == 0):
      raise ValueError("Invalid cover!")
    opt.add(Sum([subset_vars[j] for j in ind]) >= 1)
  
  ## Minimize sum of weights
  cost = Sum([weight_vars[j]*subset_vars[j] for j in range(J)])
  opt.minimize(cost)
  if not(timeout is None):
    opt.set("timeout", timeout)
  sat_check = opt.check() # actually perform the optimization
  if sat_check.r != 1:
    raise Exception("Failed to find a SAT assignment.")
  else: 
    assignment = opt.model()
    opt_ind = np.flatnonzero(np.array([assignment[subset_vars[j]].as_long() for j in range(J)]))
    return(opt_ind, opt)

## Plot solution
# opt_ind = np.flatnonzero(np.array([assignment[subset_vars[j]].as_long() for j in range(J)]))
# fig = plt.figure(figsize=(10, 4), dpi=300)
# ax = plt.gca()
# ax.scatter(*y.T, c=mean_proj, s=65, edgecolors="gray", linewidths=0.50, zorder=10)
# ax.scatter(*y[opt_ind, :].T, c="red", s=85, edgecolors="gray", linewidths=0.50, alpha=1.0, zorder=20)

# %% PyZ3 example
from z3 import *

opt = Optimize()
s1, s2, s3, s4 = Int('s1'),  Int('s2'),  Int('s3'), Int('s4')
w1, w2, w3, w4 = Real('w1'), Real('w2'), Real('w3'), Real('w4')

## Set weight values
opt.add(w1 == 2.0, w2 == 3.0, w3 == 4.0, w4 == 8.0)

## Make the integers effectively binary variables
opt.add(
  Or(s1 == 0, s1 == 1), 
  Or(s2 == 0, s2 == 1), 
  Or(s3 == 0, s3 == 1), 
  Or(s4 == 0, s4 == 1)
)

## Point-wise Constraints
opt.add(s1 + s2 >= 1, s1 + s3 >= 1, s2 + s4 >= 1, s3 + s4 >= 1)

## Minimize directly 
opt.minimize(w1*s1 + w2*s2 + w3*s3 + w4*s4)
opt.check() # sat => model will contain satisfiable assignment 
opt.model() # [s1 = 0, s2 = 1, s3 = 1, s4 = 0, w4 = 8, w3 = 4, w2 = 3, w1 = 2]

# %% 
#opt.set("timeout", 500)

# %% Matplotlib rectangle 
import matplotlib.pyplot as plt
import numpy as np

def spiral(N, sigma=1.0):
  import numpy as np
  from numpy import pi
  theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
  r_a, r_b = 2*theta + pi, -2*theta - pi
  data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T + sigma*np.random.randn(N,2)
  data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T + sigma*np.random.randn(N,2)
  return(np.vstack((data_a, data_b)))

from tallem.dimred import enclosing_radius, connected_radius
from tallem.distance import dist
from tallem.color import bin_color
# D = dist(X, as_matrix=True)
# cr, er = connected_radius(D), enclosing_radius(D)
# radius = cr + 0.05*(er-cr)
# cover, weights, tangents = neighborhood_cover(X, d=1, radius=radius)

# n, J = cover.shape
# opt = Optimize()
# weight_vars = [Real('w%f' % j) for j in range(J)]
# for j in range(J):
#   opt.add(weight_vars[j] == weights[j])
# subset_vars = [Int('s%d' % j) for j in range(J)]
# for s in subset_vars: opt.add(Or(s == 0, s == 1))
# for i in range(n):
#   ind = np.flatnonzero(cover[i,:].A)
#   if (len(ind) == 0):
#     raise ValueError("Invalid cover!")
#   opt.add(Sum([subset_vars[j] for j in ind]) >= 1)
# cost = Sum([weight_vars[j]*subset_vars[j] for j in range(J)])
# opt.minimize(cost)
# sat_check = opt.check() # actually perform the optimization
# assignment = opt.model()
# opt_ind = np.flatnonzero(np.array([assignment[subset_vars[j]].as_long() for j in range(J)]))




## Verify solution is a set cover
# import networkx as nx
# G = nx.Graph(cover)
# nx.draw(G, pos=X)

# %% Spiral

X = spiral(N=500, sigma=0.35)

fig = plt.figure(figsize=(8,5), dpi=300)

## Overall plot
cover, weights, tangents = neighborhood_cover(X, d=1, k=12)
ax = plt.subplot(1, 3, 1)
for i,x in enumerate(X): #
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c="red", alpha=0.60, linewidth=1.15, zorder=10)
_ = ax.scatter(*X.T, s=5, c=pt_col, alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
ax.set_title("Tangent space estimates", fontsize=8)
ax.axis('off')
ax.set_aspect('equal')

cover, weights, tangents = neighborhood_cover(X, d=1, k=30)
assert np.all(cover.A == cover.A.T) ## this must be true!

# best_ind, opt = weighted_set_cover_z3(cover, weights, timeout=int(10e3))
# membership = np.zeros(X.shape[0], dtype=bool)
# for si in best_ind: 
#   membership[np.flatnonzero(cover[si,:].A.flatten())] = True
# assert np.all(membership)

# best_ind_z3, opt = weighted_set_cover_z3(cover, weights, timeout=None)
# best_ind_rc2 = weighted_set_cover_rc2(cover, weights)
best_ind_greedy = greedy_weighted_set_cover(cover.shape[0], cover, weights)

from matplotlib import cm
from matplotlib.colors import rgb2hex
cpal = cm.get_cmap('viridis', 100)
col_pal = [rgb2hex(cpal(i)) for i in range(100)]

s = 3.45
pt_col = bin_color(weights, col_pal)
tangent_line_col = bin_color(weights, col_pal, scaling="equalize")

## Greedy 
ax = plt.subplot(1, 3, 2)
for i,x in zip(best_ind_greedy, X[best_ind_greedy,:]): # enumerate(X):
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c="red", alpha=0.90, linewidth=1.15, zorder=10)
_ = ax.scatter(*X.T, s=5, c=pt_col, alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
ax.set_title("WSetCover solution (RC2)")
ax.axis('off')
ax.set_aspect('equal')

ax = plt.subplot(1, 3, 3)
for i,x in zip(best_ind_greedy, X[best_ind_greedy,:]): # enumerate(X):
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c=tangent_line_col[i], alpha=0.90, linewidth=1.15, zorder=10)
ax.set_title("Tangent error")
ax.axis('off')
ax.set_aspect('equal')



ax = plt.subplot(1, 3, 2)
for i,x in zip(best_ind_z3, X[best_ind_z3,:]): # enumerate(X):
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c="red", alpha=0.90, linewidth=1.15, zorder=10)
_ = ax.scatter(*X.T, s=22, c=pt_col, alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
ax.set_title("Z3 assignment")
ax.axis('off')
ax.set_aspect('equal')

ax = plt.subplot(1, 3, 3)
for i,x in zip(best_ind_rc2, X[best_ind_rc2,:]): # enumerate(X):
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c="red", alpha=0.90, linewidth=1.15, zorder=10)
_ = ax.scatter(*X.T, s=22, c=pt_col, alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
ax.set_title("RC2 assignment")
ax.axis('off')
ax.set_aspect('equal')

# % dense spira5e3
S = spiral(N=300, sigma=0.50)
from tallem.samplers import landmarks
# X = S[landmarks(S, k=200)[0],:]
X = S
cover, weights, tangents = neighborhood_cover(X, d=1, k=15)
assert np.all(cover.A == cover.A.T) 

best_ind_rc2 = weighted_set_cover_rc2(cover, weights, adapt=True)

fig = plt.figure(figsize=(8,8), dpi=200)
ax = plt.gca()
for i,x in zip(best_ind_rc2, X[best_ind_rc2,:]): # enumerate(X):
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c="red", alpha=0.90, linewidth=1.15, zorder=10)
_ = ax.scatter(*X.T, s=22, c=bin_color(weights, col_pal), alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
ax.set_title("RC2 assignment")
ax.axis('off')
ax.set_aspect('equal')


# %% S-curve
from tallem.samplers import landmarks
from sklearn.datasets import make_s_curve, make_swiss_roll
S, p = make_s_curve(1200, noise=0.10)
# S, p = make_swiss_roll(1500)
S = S - S.mean(axis=0)
# X = S[landmarks(S, 150)[0],:]
X = S
cover, weights, tangents = neighborhood_cover(X, d=2, k=8)
weighted_set_cover_z3(cover, weights)

best_ind = greedy_weighted_set_cover(cover.shape[0], cover, weights)
# best_ind = landmarks(X, k=len(best_ind))[0]

## Plot S-curve
fig = plt.figure(figsize=(8,8), dpi=200)
ax = plt.axes(projection='3d')
ax.scatter3D(*X.T, zorder=0)
ax.scatter3D(*X[best_ind,:].T, c='red', zorder=1)

normalize = lambda x: x/np.linalg.norm(x)


from tallem.color import bin_color
cpal = cm.get_cmap('viridis', 100)
col_pal = [rgb2hex(cpal(i)) for i in range(100)]
tangent_colors = bin_color(weights, col_pal, scaling="equalize")
rel_tangent_colors = bin_color(weights[best_ind], col_pal)

## Rotate animation
import matplotlib as mpl
import matplotlib.animation as animation
n_frames = 15*5 # 15 fps * 5 to get 5 second animation


fig = plt.figure(figsize=(8,8), dpi=200) #200
ax = plt.axes(projection='3d')
ax.scatter3D(*X.T, c=tangent_colors)
ELEVATION = 15
s = 0.45
ip_iter = zip(range(len(best_ind)), best_ind, X[best_ind,:])
# ip_iter = enumerate(X)
for ri, i,x in ip_iter: # :
  north, west = tangents[i][:,0], tangents[i][:,1]
  nw = normalize(north+west)
  ne = normalize(north-west)
  sw = normalize(-north+west)
  se = normalize(-north-west)
  box = np.array([x+s*ne, x+s*se, x+s*nw, x+s*sw])
  surf = ax.plot_surface(
    box[:,0].reshape((2,2)), box[:,1].reshape((2,2)), box[:,2].reshape((2,2)),
    linewidth=0, antialiased=False, alpha=0.40, 
    color=tangent_colors[i]
    #color=rel_tangent_colors[ri]
  )
ax.axis('off')
ax.view_init(elev=ELEVATION, azim=30)

%matplotlib inline


AZIMUTH = np.linspace(0, 360, n_frames)
def update_camera(frame, *fargs):
  ax.view_init(elev=ELEVATION, azim=AZIMUTH[frame])
  ax.autoscale(enable=True, axis='x', tight=True)
  ax.autoscale(enable=True, axis='y', tight=True)
  ax.autoscale(enable=True, axis='z', tight=True)
  ax.set_facecolor('none')
anim = animation.FuncAnimation(fig, func=update_camera, frames=n_frames, interval=5000/n_frames, repeat=True)


mpl.rcParams['figure.dpi'] = 200
anim.save('s_curve_noisy_landmark.gif', writer='imagemagick', fps=15)



# %%
import numpy as np
from tallem.samplers import landmarks

X = np.random.uniform(size=(70, 2))
Lind, Lrad = landmarks(X, k = 15)
R = np.min(Lrad)

fig = plt.figure(figsize=(8,8), dpi=200)
ax = fig.gca()
plt.scatter(*X.T, c="black", zorder=1)
for i in Lind: 
  circle = plt.Circle((X[i,0], X[i,1]), R, color='yellow', alpha=0.20, zorder=0)
  ax.add_patch(circle)
plt.scatter(*X[Lind,:].T, c="red", zorder=2)
ax.axis('off')


# %% down quadratic 
import matplotlib.pyplot as plt 
import numpy as np

c = 0.75
f = lambda x: -c*(x**2)

x = np.linspace(-5, 5, 51)
y = f(x)

fig = plt.figure(figsize=(8, 3), dpi=300)
ax = fig.gca()
ax.plot(x, y, zorder=0)

x = np.linspace(-5, 5, 11)
y = f(x)
X = np.c_[x, y]
ax.scatter(*X.T, c="red", zorder=1)

from tallem.dimred import pca
_, T_x = pca(X, d=1, coords=False)
T_x = np.flipud(T_x)
proj_points = (X @ T_x)*T_x.T

P = proj_points[2:-2,:]

ax.scatter(*P.T, c="blue", zorder=2)
for i, p in enumerate(P):
  dy = np.linalg.norm(p - X[i+2,:])
  ax.arrow(x=X[i+2,0],y=X[i+2,1],dx=0,dy=dy-0.45, zorder=0, head_width=0.05, head_length=0.40, length_includes_head=True)
ax.plot(P[:,0]*1.15, P[:,1], c="green", alpha=0.50, zorder=0)
ax.axis('off')