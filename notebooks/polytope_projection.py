# %% simplex projectors 
import matplotlib.pyplot as plt
from tallem.datasets import * 
from tallem.polytope import *
from scipy.spatial import ConvexHull, Delaunay

X = np.random.uniform(size=(20,2), low = 5, high = 10)
Y = np.random.uniform(size=(210,2), low = 4, high = 11)

fig, ax = plt.subplots(figsize=(8,8))
plt.triplot(X[:,0], X[:,1], Delaunay(X).simplices)
ax.scatter(*X.T, color="blue") 
ax.scatter(*Y.T, color="green")
Z = np.array([project_hull(y, X) for y in Y])
ax.scatter(*Z.T, color="purple")
for y,z in zip(Y,Z): plt.plot(*np.vstack((y,z)).T, color="purple")

## Level sets
hull = ConvexHull(X)
barycenter = np.mean(hull.points[hull.vertices], axis=0)
ray = lambda x: (x - barycenter)/np.linalg.norm(x - barycenter)
Z = np.array([project_ray(x, ray(x), hull) for x in X])

fig, ax = plt.subplots(figsize=(8,8))
plt.triplot(X[:,0], X[:,1], Delaunay(X).simplices)
ax.scatter(*barycenter, color="red") 
ax.scatter(*X.T, color="blue") 
ax.scatter(*Y.T, color="green")
ax.scatter(*Z.T, color="purple")
for y,z in zip(Y,Z): plt.plot(*np.vstack((y,z)).T, color="purple")
#sdist_to_boundary(Y, hull, method="ray")


fig, ax = plt.subplots(figsize=(8,8))
plt.triplot(X[:,0], X[:,1], Delaunay(X).simplices)
ax.scatter(*barycenter, color="red") 
ax.scatter(*X.T, color="blue") 
ax.scatter(*Y.T, color="green")

sdist_to_boundary(Y, hull, method="ray")


#### DEBUG 

u = barycenter - Y[0,:]
u = u / np.linalg.norm(u)
z = project_ray(Y[0,:], u, hull)

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		# raise RuntimeError("no intersection or line is within plane")
		return(np.repeat(np.inf, len(planePoint)))
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi, si


u = Y[0,:] - barycenter
u = u / np.linalg.norm(u)

u = barycenter - Y[0,:]
u = u / np.linalg.norm(u)

Z = []
for f in range(hull.equations.shape[0]):
	n = hull.equations[f,:-1]
	Q = hull.points[hull.vertices]
	facet = hull.points[hull.simplices[f,:],:]
	p0 = np.mean(facet, axis=0)
	z, d = LinePlaneCollision(n, p0, u, Y[0,:])
	if not(in_hull(z, hull)):
		z = np.repeat(-np.inf, Y.shape[1])
	Z.append(z)
Z = np.array(Z)
z = Z[np.argmin(np.linalg.norm(Y[0,:] - Z, axis = 1)),:]


Z = []
for y in Y:
	u = y - barycenter
	u = u / np.linalg.norm(u)
	z = project_ray(y, u, hull)
	Z.append(z)
Z = np.array(Z)


sdist_to_boundary(Y, hull, method="ray")


## Show contour level sets
X = np.random.uniform(size=(50,2), low = 5, high = 10)
# X = np.array([[-1,1], [-1,-1],[10,-1], [10,1]])
hull = ConvexHull(X)
Q = hull.points[hull.vertices]

## Countour
min_x, min_y = np.min(hull.points, axis = 0)
max_x, max_y = np.max(hull.points, axis = 0)
MX, MY = np.meshgrid(np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100))
XY = np.c_[np.ravel(MX), np.ravel(MY)]
sd = sdist_to_boundary(XY, hull, method="orthogonal")

sd = sdist_to_boundary(XY, hull, method="ray")
dist_to_center = np.linalg.norm(XY - np.mean(hull.points[hull.vertices], axis = 0), axis = 1)
nd = sd/(dist_to_center+sd)

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.contourf(MX, MY, nd.reshape(MX.shape), levels=100)
plt.plot(*np.vstack((Q[0:Q.shape[0],:], Q[0,:])).T, color="red")
plt.gca().set_aspect('equal')

# fig, ax = plt.subplots(figsize=(8,8))
# plt.plot(*np.vstack((Q[0:Q.shape[0],:], Q[0,:])).T)
# ax.scatter(*barycenter, color="red") 
# ax.scatter(*X.T, color="blue") 
# ax.scatter(*Y.T, color="green")
# ax.scatter(*Z.T, color="purple")
# for y,z in zip(Y, Z): plt.plot(*np.vstack((y,z)).T, color="purple")
# plt.gca().set_aspect('equal')





# ax.scatter(*project_ray(Y[0,:], u, hull), color="orange")

########

#V = np.random.uniform(size=(5,2), low=0.0, high=1.0)
V = np.array([[3,7],[10,7],[8,2],[5,3]]) 

alpha = np.mean(V, axis=0) # barycenter 
x = alpha + np.random.uniform(size=2)

hull = ConvexHull(V)
Q = hull.points[hull.vertices]
#plt.triplot(V[:,0], V[:,1], Delaunay(V).simplices)
# hull.equations[1,-1]*np.linalg.norm(np.array([5,-2]))

plt.plot(*np.vstack((Q[0:Q.shape[0],:], Q[0,:])).T)
plt.scatter(*alpha, color="green")

X = np.random.uniform(size=(80,2), low=-1.0, high=1.0)+alpha
Z = np.array([project_from_center(x, hull) for x in X])

plt.scatter(*X.T, color="red")
plt.scatter(*Z.T, color="orange")
for x,z in zip(X, Z):
	plt.plot(*np.vstack((x,z)).T, color="purple")
plt.scatter(*alpha, color="green")

from tallem.polytope import *

Z_int = project_hull(X, V, method="interior")
Z_ext = project_hull(X, V, method="complement")
Z_bou = project_hull(X, V, method="boundary")

plt.plot(*np.vstack((Q[0:Q.shape[0],:], Q[0,:])).T)
plt.scatter(*alpha, color="green")
plt.scatter(*Z_int.T, color="purple")
plt.scatter(*Z_ext.T, color="orange")

plt.scatter(*Z_bou.T, color="blue")


plt.scatter(*z, color="orange")
# plt.scatter(*Q.T, c=dist)

p0 = np.mean(hull.points[hull.simplices[1,:]], axis = 0)
plt.scatter(*p0, color="purple")

plt.scatter(*z, color="orange")
plt.plot(hull.equations[0,:-1])


# import potpourri3d as pp3d
# delh = Delaunay(ConvexHull(points=V).points)
# plt.triplot(V[:,0], V[:,1], delh.simplices)
# solver = pp3d.MeshHeatMethodDistanceSolver(np.c_[V, np.zeros(V.shape[0])],delh.simplices)
# dist = solver.compute_distance(0)

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		# raise RuntimeError("no intersection or line is within plane")
		return(np.repeat(np.inf, len(planePoint)))
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi, si

	# ndotu = planeNormal.dot(rayDirection)
	# if abs(ndotu) < epsilon:
	# 	# raise RuntimeError("no intersection or line is within plane")
	# 	return(np.repeat(np.inf, len(planePoint)))
	# w = rayPoint - planePoint
	# si = -planeNormal.dot(w) / ndotu
	# Psi = w + si * rayDirection + planePoint
	# return Psi, si

A, B = hull.points[hull.simplices[0,:]]
z = LinePlaneCollision(n, 0.5*(A + B) + b, u, alpha)

p0 = 0.5*(A + B)
b = hull.equations[0,-1]
u = (x - alpha)
u = u / np.linalg.norm(u)
n = hull.equations[0,:-1]
d = np.dot(p0 - alpha, n)/np.dot(u, n)
z = alpha - u*d

np.dot(hull.equations[:,0:2], alpha.reshape((2,1))).flatten() < hull.equations[:,2]

t = (hull.equations[0,-1] - np.dot(n, alpha))/(np.dot(n, x - alpha))
d = -np.dot(n, p0 - b)

def proj_line_seg(X, x0, bary=False):
	''' Projects point x0 onto line segment X=(x1, x2) where X == (d x 2) matrix defining the line segment'''
	x1, x2 = X[:,0], X[:,1]
	alpha = float(np.dot(np.transpose(x1-x2), x0-x2))/(np.dot(np.transpose(x1-x2), x1-x2))
	alpha = max(0,min(1,alpha))
	y = alpha*x1 + (1-alpha)*x2
	theta = np.array([alpha, 1-alpha])
	return(theta if bary else y)



X = np.random.uniform(size=(20,2), low = 5, high = 10)
Y = np.random.uniform(size=(50,2), low = 4, high = 11)
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
from scipy.spatial.qhull import _Qhull
# wut = _Qhull(b"i", X, options=b"Qw QG18 QG19",furthest_site=False, incremental=False, interior_point=None)

hull = ConvexHull(points=X)
delh = Delaunay(hull.points)
delh.find_simplex(np.vstack((X, Y)))




from scipy.spatial import ConvexHull
from quadprog import solve_qp

# From: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(points, queries):
	from scipy.spatial.qhull import _Qhull
	hull = _Qhull(b"i", points, options=b"", furthest_site=False, incremental=False, interior_point=None)
	equations = hull.get_simplex_facet_array()[2].T
	return np.all(queries @ equations[:-1] < - equations[-1], axis=1)

def in_hull(points, x):
	from scipy.optimize import linprog
	n_points, n_dim = len(points),len(x)
	c = np.zeros(n_points)
	A = np.r_[points.T,np.ones((1,n_points))]
	b = np.r_[x, np.ones(1)]
	lp = linprog(c, A_eq=A, b_eq=b)
	return lp.success

def point_in_hull(point, hull, tolerance=1e-12):
	return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

Z = np.array([proj2hull(y, hull.equations) for y in Y])


fig = plt.figure(figsize=(16,16))
plt.triplot(X[:,0], X[:,1], delh.simplices)
plt.scatter(*X.T, color="blue")
plt.scatter(*Y.T, color="red")
# plt.scatter(*Z.T, color="green")
# for y,z in zip(Y, Z):
# 	plt.plot(*np.vstack((y,z)).T, color="purple")

hull_vertices = hull.points[hull.vertices,:]

Z = np.array([proj_line_seg(hull_vertices[0:2,:].T, y) for y in Y])
plt.scatter(*Z.T, color="green")
for y,z in zip(Y, Z):
	plt.plot(*np.vstack((y,z)).T, color="purple")

hull.points


## Testing line projection
fig = plt.figure(figsize=(12,12))
plt.triplot(X[:,0], X[:,1], delh.simplices)
plt.scatter(*X.T, color="blue")
plt.scatter(*Y.T, color="red")

db, Z = dist_to_boundary(Y, hull)
plt.scatter(*Z.T, color="purple") 
for y,z in zip(Y, Z):
	plt.plot(*np.vstack((y,z)).T, color="purple")



hull_vertices = X[hull.vertices,:]

fig = plt.figure(figsize=(8,8))
convex_hull_plot_2d(hull, ax=ax)

hull.equations[:-1]
np.all(Y @ hull.equations[:-1] < -hull.equations[-1], axis=1)



# plot_in_hull(X, ConvexHull(points=X))

def plot_in_hull(p, hull):
    """
    plot relative to `in_hull` for 2d data
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection

    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    # plot triangulation
    poly = PolyCollection(hull.points[hull.vertices], facecolors='w', edgecolors='b')
    plt.clf()
    plt.title('in hull')
    plt.gca().add_collection(poly)
    plt.plot(hull.points[:,0], hull.points[:,1], 'o', hold=1)


    # plot the convex hull
    edges = set()
    edge_points = []

    def add_edge(i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(hull.points[ [i, j] ])

    for ia, ib in hull.convex_hull:
        add_edge(ia, ib)

    lines = LineCollection(edge_points, color='g')
    plt.gca().add_collection(lines)
    plt.show()    

    # plot tested points `p` - black are inside hull, red outside
    inside = in_hull(p,hull)
    plt.plot(p[ inside,0],p[ inside,1],'.k')
    plt.plot(p[-inside,0],p[-inside,1],'.r')


# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))


# %% CVXOPT
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import solvers, matrix
# minimize    (1/2)*x'*P*x + q'*x
#         subject to  G*x <= h
#                     A*x = b.

x_outer = np.array([4.1,6.0])
x_inner = np.array([2.1,4.0])
P = np.array([[0, 0], [3,0], [4,5], [2,6], [0,4]])

plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*x_outer, c = 'red')

from pypoman import compute_polytope_halfspaces
q = -np.reshape(x_outer, (2,1)) 
A, b = compute_polytope_halfspaces(P)
b = b.reshape((len(b), 1))
sol = solvers.qp(P=matrix(np.eye(2)), q=matrix(q), G=matrix(A), h=matrix(b))

y = np.asarray(sol['x'])
plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*y, c = 'red')

q = -np.reshape(x_inner, (2,1)) 
sol = solvers.qp(P=matrix(np.eye(2)), q=matrix(q), G=matrix(-A), h=matrix(-b))

# %% 
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([2.1,4.0])
# x0 = np.array([0.5, 5.2])
x0 = np.array([3.5, 6.2])
P = np.array([[0, 0], [3,0], [4,5], [2,6], [0,4]])

plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*x0, c = 'red')

# plt.plot(*(np.vstack([(P-x0), (P-x0)[0,:]])).T)
# plt.scatter(*(x0 - x0), c = 'red')


# %% QP attempt 5
I = np.ones((2,1))
y = solve_qp(P=np.array(P @ P, dtype=np.float64), q=x0, A=I.T, b=np.array([1.0]), verbose=True, solver="cvxopt")


#%% Quadratic programming reduction 
import numpy as np
from numpy import array, dot
from qpsolvers import solve_qp
from pypoman import compute_polytope_halfspaces

q = -np.reshape(x0, (2,1)) 
A, b = compute_polytope_halfspaces(P)
b = b.reshape((len(b), 1))

y = solve_qp(P=np.eye(2), q=q, G=A, h=b, verbose=True, solver="cvxopt")
y = solve_qp(P=np.eye(2), q=q, G=-A, h=-b, verbose=True, solver="quadprog")
#y = solve_qp(P=np.eye(2), q=q, A=A, b=b, verbose=True, solver="cvxopt")
print("QP solution: y = {}".format(y))

# x0 - np.dot(A.T, np.linalg.lstsq(A.T, x0,rcond=None)[0])

# %% Attempt 4 on the QP
# https://stackoverflow.com/questions/43850745/why-does-cvxopt-give-a-rank-error-for-this-nonlinear-network-flow-optimisation
u, s, vt = np.linalg.svd(A, compute_uv=True, full_matrices=False)
A_min, b_min = np.diag(s) @ vt, np.reshape(u.T @ b, (u.shape[1], 1))
y = solve_qp(P=np.eye(2), q=q, A=A_min, b=b_min, verbose=True, solver="cvxopt")

# %% Lest squares soln 
Q = P - x0
At, bt = compute_polytope_halfspaces(Q)
u, s, vt = np.linalg.svd(At, compute_uv=True, full_matrices=False)
#vt.T @ np.diag(1.0/s) @ u[:,0:2].T
# z = np.linalg.pinv(A) @ b

z = (vt.T @ np.diag(1.0/s) @ u.T) @ bt.reshape((len(bt), 1))

#z = np.linalg.pinv(At) @ bt
#z = x0 + (np.linalg.pinv(At) @ bt)

plt.plot(*(np.vstack([Q, Q[0,:]])).T)
plt.scatter(*(x0 - x0), c = 'red')
plt.scatter(*z, c="orange")



#%% 
plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*x0, c = 'red')
plt.scatter(*y, c = 'green')
plt.arrow(*x0.T, *(y-x0).T, width=0.025)
# plt.scatter(*z, c = 'purple')

# %% Hammer it
from scipy.optimize import minimize, LinearConstraint
dist = lambda x: np.linalg.norm(x - x0)
# on_boundary = LinearConstraint(A, lb=np.zeros(b.shape), ub=b, keep_feasible=False)
on_boundary = LinearConstraint(-A, lb=np.full(b.shape, -np.inf).flatten(), ub=-b.flatten(), keep_feasible=False)
minimize(dist, x0=x0, constraints=on_boundary)

z = minimize(dist, x0=x0, constraints=on_boundary)['x']

#%% 
from numpy.typing import ArrayLike
def project_to_polytope(X: ArrayLike, P: ArrayLike):
	X, P = np.asanyarray(X), np.asanyarray(P)
	assert X.ndim == P.ndim and X.shape[1] == P.shape[1], "X and P must have the same number of columns"
	n, d = X.shape
	A, b = compute_polytope_halfspaces(P)
	# A = ConvexHull(P).equations
	# b = A[:,-1]
	#A = A[:,:-1]
	b.reshape((len(b), 1))
	Q = np.eye(d)               # x^T Q x 
	out = np.zeros(shape=(n, d), dtype=np.float32)
	for i, x0 in enumerate(X): 
		q = -np.reshape(x0, (d,1)) # q^T x 
		y = solve_qp(P=Q, q=q, G=A, h=b, verbose=False, solver="cvxopt")
		if y is None:
			out[i,:] = np.repeat(np.nan, d)
		else:
			out[i,:] = y
	return(out)
	
#%% 
bbox_min = np.apply_along_axis(np.min, axis=0, arr=P)
bbox_max = np.apply_along_axis(np.max, axis=0, arr=P)

xr = np.random.uniform(bbox_min[0]-2.50,bbox_max[0]*1.50,35)
yr = np.random.uniform(bbox_min[1]-2.50,bbox_max[1]*1.50,35)
X = np.c_[xr, yr]
Y = project_to_polytope(X, P)

#%% 
plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([P, P[0,:]])).T)
for x0, y in zip(X, Y):
	plt.scatter(*x0, c = 'red')
	plt.scatter(*y, c = 'green')
	plt.arrow(*x0.T, *(y-x0).T, width=0.025)
c = np.mean(P, axis=0)
plt.scatter(*c, c = 'orange')

# %% 
from scipy.spatial import ConvexHull
hull = ConvexHull(P)

# Get points on the inside
# dx = [np.linalg.norm(y - x) for x,y in zip(X, Y)]

# S_projects = [project_to_polytope(X, P[s,:]) for s in hull.simplices]
# diffs = [np.linalg.norm(X - S, axis = 1) for S in S_projects]
# min_facet = np.argmin(np.vstack(diffs), axis = 0)

# facet_projections = np.c_[[S_projects[m_index][i,:] for i, m_index in enumerate(min_facet)]]

A = P 
Q = A.T @ A
p = -0.5*A.T @ x0.reshape((2,1))

facet_projections = project_to_polytope(X, P[hull.simplices[2],:])

# %% 
plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([x, x[0,:]])).T)
for x0, y in zip(X, facet_projections):
	plt.scatter(*x0, c = 'red')
	plt.scatter(*y, c = 'green')
	plt.arrow(*x0.T, *(y-x0).T, width=0.025)


# %% Projection w/ QP: attempt #2
y = np.array([0.5, 5.2])
P = np.array([[0, 0], [3,0], [4,5], [2,6], [0,4]])
Q = P - y
e = np.ones((len(y),1))
q = np.zeros((len(y), 1))
G = np.zeros((len(y), len(y)))
h = np.zeros((len(y), 1))
y = solve_qp(
	P=Q.T, q=q, A=e, b=np.array([1]), G=G, h=h, 
	lb=np.zeros((len(y), 1)), ub=np.zeros((len(y), 1)), 
	verbose=True, 
	solver="quadprog")


np.vstack([A, G])
# \mbox{minimize} \frac{1}{2} x^T P x + q^T x \\
# \mbox{subject to}
# 		& G x \leq h                \\
# 		& A x = b                    \\
# 		& lb \leq x \leq ub


#%%
import cdd

# %% 
from pypoman import compute_polytope_halfspaces
A, b = compute_polytope_halfspaces(x)

from numpy import array, eye, ones, vstack, zeros
from pypoman import plot_polygon, project_polytope
n = 10  # dimension of the original polytope
p = 2   # dimension of the projected polytope

# Original polytope:
# - inequality constraints: \forall i, |x_i| <= 1
# - equality constraint: sum_i x_i = 0
A = vstack([+eye(n), -eye(n)])
b = ones(2 * n)
C = ones(n).reshape((1, n))
d = array([0])
ineq =   # A * x <= b
eq = (C, d)    # C * x == d

# Projection is proj(x) = [x_0 x_1]
E = zeros((p, n))
E[0, 0] = 1.
E[1, 1] = 1.
f = zeros(p)
proj = (E, f)  # proj(x) = E * x + f

vertices = project_polytope(proj, ineq=(A, b), eq=eq, method='bretl')


# Chebyshev center of a polyhedron
# the  Farthest point furthest away from all inequalities.


# %% Wolfe's algorithm 

# %% 
import matplotlib.pyplot as plt
import numpy as np

x_outer = np.array([4.1,6.0])
x_inner = np.array([2.1,4.0])
P = np.array([[0, 0], [3,0], [4,5], [2,6], [0,4]])



# Barycenter
c = np.mean(P, axis = 0)

plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*x_inner, c = 'red')
plt.scatter(*(pt + x_outer), c = 'purple')
plt.scatter(*c, c = 'blue')


u = (x_inner - c)/(np.linalg.norm(x_inner - c))

from itertools import combinations
diam = np.max([np.linalg.norm(P[i,:] - P[j,:]) for i, j in combinations(range(P.shape[0]), 2)])

Q = []
for p in np.linspace(0.10, 1.50, 15):
	x_outer = c + u*(diam*p)
	pt, bary = wolfe_proj(P-x_outer)
	Q.append(bary @ P)


plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*x_outer, c = 'red')
plt.scatter(*c, c = 'blue')
plt.scatter(*(np.c_[Q]).T, c = 'green')








