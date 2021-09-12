# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

# %%
import matplotlib.pyplot as plt 
import numpy as np
import numpy.typing as npt

mu = [8, 8]
Sigma = np.diag([2,2])
locations = np.array([[0,0], [0,1], [1,0], [1,1]])


def mvn_density(locations, mu, Sigma):
	d, k, S_inv, mu = np.linalg.det(Sigma), len(mu), np.linalg.inv(Sigma), np.asanyarray(mu)[:,np.newaxis]
	result = np.zeros(locations.shape[0])
	for i, x in enumerate(locations): 
		x = x.reshape(mu.shape)
		f_x = np.exp(-0.5*((x - mu).T @ S_inv @ (x - mu)))/np.sqrt((2*np.pi)**k * d)
		result[i] = f_x
	return(result)
	
def gaussian_pixel(center, Sigma, d=17,s=1):
	from src.tallem.utility import cartesian_product
	grid_ind = cartesian_product([range(d), range(d)])
	grid = np.zeros(shape=(d,d))	
	F = mvn_density(grid_ind, center, Sigma)
	for i, ind in enumerate(grid_ind):
		r,c = ind
		grid[r,c] = s*F[i]
	return(grid)

p = gaussian_pixel([8.5, 8.5], Sigma, s=1)
plt.imshow(p, cmap='gray', vmin=0, vmax=np.max(p))

# %% theta and r 
r = np.random.uniform(low=0, high=1, size=1500)
theta = np.random.uniform(low=0, high=2*np.pi, size=1500)
X = r * np.cos(theta)
Y = r * np.sin(theta)
P = [np.ravel(gaussian_pixel([x,y], Sigma, s=1)) for x,y in zip(X, Y)]

R = np.linspace(0, 1, 300)
B = [np.ravel(gaussian_pixel([17/2, 17/2], Sigma, s=r)) for r in R]

N = np.vstack((np.asanyarray(P), np.asanyarray(B)))




# %%
max_val = mvn_density(np.array([[0,0]]), [0, 0], Sigma)[0]
phi = np.random.uniform(low=0, high=2*np.pi, size=100)
theta = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=100)

X, Y = (phi/(2*np.pi))*17, ((theta/np.pi)*17) + (17/2)
# print(np.min(X), np.max(X), np.min(Y), np.max(Y))

P = [gaussian_pixel([x, y], Sigma, s=1) for x,y in np.c_[X,Y]]

#print((phi/(2*np.pi)))
# plt.imshow(P[1], cmap='gray', vmin=0, vmax=np.max(P[i]))

fig = plt.figure(figsize=(8, 8))
for i, p in enumerate(P[0:30]):
	fig.add_subplot(5, 10, i+1)
	plt.imshow(P[i], cmap='gray', vmin=0, vmax=max_val)
	fig.gca().axes.get_xaxis().set_visible(False)
	fig.gca().axes.get_yaxis().set_visible(False)


## Generate images w/ varying intensity 
phi = np.repeat(np.pi, 100)
theta = np.repeat(0.0, 100)
R = np.random.uniform(low=0,high=1,size=100)
X, Y = (phi/(2*np.pi))*17, ((theta/np.pi)*17) + (17/2)
Q = [gaussian_pixel([x, y], Sigma, s=np.abs(r)) for x,y,r in np.c_[X,Y,R]]

phi = np.repeat(0, 100)
R = np.random.uniform(low=-1,high=0,size=100)
X, Y = (phi/(2*np.pi))*17, ((theta/np.pi)*17) + (17/2)
Q2 = [gaussian_pixel([x, y], Sigma, s=np.abs(r)) for x,y,r in np.c_[X,Y,R]]


for i, p in enumerate(Q[0:20]):
	fig.add_subplot(5, 10, i+1+30)
	plt.imshow(Q[i], cmap='gray', vmin=0, vmax=max_val)
	fig.gca().axes.get_xaxis().set_visible(False)
	fig.gca().axes.get_yaxis().set_visible(False)

# 	# axs[i].imshow(P[i], cmap='gray', vmin=0, vmax=np.max(P[i]))
# 	# axs.gca().axes.get_yaxis().set_visible(False)
# 	# axs.xticks([])
# 	# axs.yticks([])


# %%
sphere = np.vstack([np.ravel(p) for p in P])
line_s1 = np.vstack([np.ravel(q) for q in Q])
line_s2 = np.vstack([np.ravel(q) for q in Q2])
dot = np.vstack((sphere, line_s1, line_s2))
dot.shape

# %% Need geodesic distance on sphere
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
S2 = Hypersphere(2)

pt0, pt1 = np.array([1,0,0]), np.array([0,0,1])
S2.metric.dist(pt0, pt1)

# Confirm extrinsic coordinates were used
# np.apply_along_axis(np.linalg.norm, 1, S2.random_uniform(10))

P_ext = S2.random_uniform(1000)
P_int = S2.extrinsic_to_intrinsic_coords(P_ext) # [-1, 1], 


# %% plot the hypersphere 
import geomstats.backend as gs
import geomstats.visualization as vis
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 15))
ax = vis.plot(P_ext, space="S2")
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

Z = np.zeros((100, 3))
Z[:,2] = np.random.uniform(low=-1, high=1, size=100)
ax.scatter(*Z.T)

# %% Get f: X -> B  + geodesic distance 
S2.extrinsic_to_intrinsic_coords(Z) # surprisingly this works

# Geodesic distance should just be piecewise function 
points_ext = np.vstack((P_ext, Z))

def geodesic_dist(x, y):
	x, y = np.asanyarray(x), np.asanyarray(y)
	xr, yr = np.linalg.norm(x), np.linalg.norm(y)
	if xr == 1 and yr == 1:
		return(S2.metric.dist(x,y))
	elif xr == 1 and yr != 1: 
		r_dist = np.abs(y[2])
		tip = np.array([0,0,-1]) if y[2] <= 0 else np.array([0,0,1])
		return(S2.metric.dist(x,tip) + r_dist)
	elif xr != 1 and yr == 1:
		r_dist = np.abs(x[2])
		tip = np.array([0,0,-1]) if x[2] <= 0 else np.array([0,0,1])
		return(S2.metric.dist(y,tip) + r_dist)
	else:
		return(np.linalg.norm(x-y))

# Make sure distance function is working fine! 
# from src.tallem.distance import dist
# D = dist(points_ext, metric=geodesic_dist)


from src.tallem.landmark import landmarks, landmarks_dist
#L = landmarks(points_ext, k = 25, metric = geodesic_dist)
L = landmarks_dist(points_ext, k = 25, metric = geodesic_dist)

# %% Visualize the landmarks

fig = plt.figure(figsize=(15, 15))
ax = vis.plot(P_ext, space="S2", color="blue")
ax.scatter(*Z.T, color="blue")
ax.scatter(*points_ext[L['indices'], :].T, color="red", s=120)
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

# %% Use geodesic distance + landmarks to make landmark 'BallCover' 
from src.tallem.cover import BallCover

centers = points_ext[L['indices'], :]
r = np.min(L['radii'])

cover = BallCover(centers = centers, radii = r, metric = geodesic_dist)

# %% 
cover.construct(points_ext)

# %% 
from src.tallem.cover import partition_of_unity
cover.construct(N)
pou = partition_of_unity(N, cover)

# D = dist(points_ext, centers, metric=geodesic_dist)
# np.sum(D[:,0] <= r) # 4 

# np.sum(cover.set_distance(points_ext, 0) <= 1.0)

# np.maximum(0.0, 1.0 - cover.set_distance(points_ext, 0)).nonzero()

# %%
from src.tallem import TALLEM
from src.tallem.cover import BallCover
from src.tallem.landmark import landmarks 


L = landmarks(N, k=25)
r = np.min(L['radii'])
centers = N[L['indices'],:]
cover = BallCover(centers, r)
cover.construct(N)

## construct the cover
#cover.construct(dot)

# for k,v in cover.items():
# 	print(v)

# print(cover.centers)
## Construct the partition of unity 

# LandmarkCover()

## Construct a cover over the polar coordinate
#cover = IntervalCover(B[:,[1]], n_sets = 10, overlap = 0.30, gluing=[1])

## Local euclidean models are specified with a function
# f = lambda x: classical_MDS(dist(x, as_matrix=True), k = 2)

# ## Parameterize TALLEM + transform the data to the obtain the coordinization
# embedding = TALLEM(cover=cover, local_map=f, n_components=3)
# X_transformed = embedding.fit_transform(X, B_polar)


# %% intrinsic point coordinates (wrong)
# from src.tallem.dimred import cmds
# from src.tallem.distance import dist

# f = lambda x: cmds(dist(x, as_matrix=True), 2)
# embedding = TALLEM(cover, local_map=f, n_components=3)
# coords = embedding.fit_transform(points_ext, B=points_ext)


# %% Converting between intrinsic and extrinsic point coordinates
from src.tallem.dimred import cmds
from src.tallem.distance import dist

Sigma = np.diag([1,1])
int_pts = S2.extrinsic_to_intrinsic_coords(points_ext)
Z = np.zeros((int_pts.shape[0], 17*17))
for i, pt in enumerate(int_pts):
	if pt[0] != 0: 
		z = np.ravel(gaussian_pixel(((pt + 1)/2)*17, Sigma, s=1))
	else: 
		z = np.ravel(gaussian_pixel([17/2, 17/2], Sigma, s=3*np.abs(pt[1]))) 
	Z[i,:] = z

# p = gaussian_pixel([17/2, 17/2], Sigma, s=3*np.abs(pt[1]))
# plt.imshow(p, cmap='gray', vmin=0, vmax=np.max(p))

# %% 
f = lambda x: cmds(dist(x, as_matrix=True), 3)

from src.tallem.cover import BallCover, IntervalCover
L = landmarks(N, k=10)
r = np.min(L['radii'])
centers = N[L['indices'],:]
cover = BallCover(centers, r)

#%%
from src.tallem.dimred import cmds
from src.tallem.distance import dist
f = lambda x: cmds(dist(x, as_matrix=True), 3)
embedding = TALLEM(cover, local_map=f, n_components=3)
coords = embedding.fit_transform(N, B=N)

## one of the local models is 2 dimensional
[m.shape for m in embedding.models.values()]

# %%
plt.scatter(*coords.T) # 2d 

# %% 3d plot 
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(projection='3d')
ax.scatter(*coords.T)



#%% isomap 
import numpy as np
from src.tallem.dimred import *
x = np.random.uniform(size=(100,4))

#%% 
G = neighborhood_graph(x, radius = 2.5)

geodesic_dist(G.todense())

neighborhood_list(x, x, raidus = 2.5)

geodesic_dist(neighborhood_graph(x, radius = 2.5).todense())
isomap(x, d=2, radius=2.5)
