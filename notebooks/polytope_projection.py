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

# From: https://github.com/kboulif/Non-negative-matrix-factorization-NMF-/blob/648ea6026a6ef11919f87885fd16ffebf51e5e28/NMF.py
def wolfe_proj(X,epsilon=1e-6,threshold=1e-8,niter=100,verbose=False):
	''' 
	Projects origin onto the convex hull of the rows of 'X' using Wolfe method. The algorithm is by Wolfe in his paper 'Finding the nearest point in A polytope'. 
	Parameters:	
		'epsilon', 'threshold': Algorithm parameters determining approximation acceptance thresholds. These parameters are denoted as (Z2,Z3) and Z1, in the main paper, respectively. Default values = 1e-6, 1e-8.
		'niter': Maximum number of iterations. Default = 10000.	
		'verbose': If set to be True, the algorithm prints out the current set of weights, active set, current estimate of the projection after each iteration. Default = False.
	'''
	n = X.shape[0]
	d = X.shape[1]
	max_norms = np.min(np.sum(np.abs(X)**2,axis=-1)**(1./2))
	s_ind = np.array([np.argmin(np.sum(np.abs(X)**2,axis=-1)**(1./2))])
	w = np.array([1.0])
	E = np.array([[-max_norms**2, 1.0], [1.0, 0.0]])
	isoptimal = 0
	iter = 0
	while (isoptimal == 0) and (iter <= niter):
			isoptimal_aff = 0
			iter = iter+1
			P = np.dot(w,np.reshape(X[s_ind,:], (len(s_ind), d)))
			new_ind = np.argmin(np.dot(P,X.T))
			max_norms = max(max_norms, np.sum(np.abs(X[new_ind,:])**2))
			if (np.dot(P, X[new_ind,:]) > np.dot(P,P) - threshold*max_norms):
					isoptimal = 1
			elif (np.any(s_ind == new_ind)):
					isoptimal = 1
			else:
					y = np.append(1,np.dot(X[s_ind,:], X[new_ind,:]))
					Y = np.dot(E, y)
					t = np.dot(X[new_ind,:], X[new_ind,:]) - np.dot(y, np.dot(E, y))
					s_ind = np.append(s_ind, new_ind)
					w = np.append(w, 0.0)
					E = np.block([[E + np.outer(Y, Y)/(t+0.0), -np.reshape(Y/(t+0.0), (len(Y),1))], [-Y/(t+0.0), 1.0/(t+0.0)]])
					while (isoptimal_aff == 0):
							v = np.dot(E,np.block([1, np.zeros(len(s_ind))]))
							v = v[1:len(v)]          
							if (np.all(v>epsilon)):
									w = v
									isoptimal_aff = 1
							else:
									POS = np.where((w-v)>epsilon)[0]
									if (POS.size==0):
											theta = 1
									else:
											fracs = (w+0.0)/(w-v)
											theta = min(1, np.min(fracs[POS]))
									w = theta*v + (1-theta)*w
									w[w<epsilon] = 0
									if np.any(w==0):
											remov_ind = np.where(w==0)[0][0]
											w = np.delete(w, remov_ind)
											s_ind = np.delete(s_ind, remov_ind)
											col = E[:, remov_ind+1]
											E = E - (np.outer(col,col)+0.0)/col[remov_ind+1]
											E = np.delete(np.delete(E, remov_ind+1, axis=0), remov_ind+1, axis=1)
			
			y = np.dot(X[s_ind,:].T, w)
			if (verbose==True):
					print ('X_s=')
					print (X[s_ind,:])
					print ('w=')
					print (w)
					print ('y=')
					print (y)
					print ('s_ind=')
					print (s_ind)

			weights = np.zeros(n)
			weights[s_ind] = w
	return [y, weights]

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


