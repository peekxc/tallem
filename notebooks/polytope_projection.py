# %% Patch the PYTHONPATH to run scripts native to parent-level folder
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

# %% 
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([0.5, 5.2])
P = np.array([[0, 0], [3,0], [4,5], [2,6], [0,4]])

plt.plot(*(np.vstack([P, P[0,:]])).T)
plt.scatter(*x0, c = 'red')


#%% Quadratic programming reduction 
from numpy import array, dot
from qpsolvers import solve_qp
from pypoman import compute_polytope_halfspaces

q = -np.reshape(x0, (2,1)) 
A, b = compute_polytope_halfspaces(P)
y = solve_qp(P=np.eye(2), q=q, G=A, h=b.reshape((len(b), 1)), verbose=True, solver="cvxopt")
print("QP solution: y = {}".format(y))

# x0 - np.dot(A.T, np.linalg.lstsq(A.T, x0,rcond=None)[0])

#%% 
plt.gca().set_aspect('equal')
plt.plot(*(np.vstack([x, x[0,:]])).T)
plt.scatter(*x0, c = 'red')
plt.scatter(*y, c = 'green')
plt.arrow(*x0.T, *(y-x0).T, width=0.025)

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
plt.plot(*(np.vstack([x, x[0,:]])).T)
for x0, y in zip(X, Y):
	plt.scatter(*x0, c = 'red')
	plt.scatter(*y, c = 'green')
	plt.arrow(*x0.T, *(y-x0).T, width=0.025)


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



