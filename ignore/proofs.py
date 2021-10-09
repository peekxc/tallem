import numpy as np

J, N = 50, 1000

## Proof that the initial guess can be computed quickly
## A ~ (J x N) => svd takes O(JNR) <= O(J^2 N) time when R = J, J < N
# %% SVD 
%%time
A = np.random.uniform(size=(J,N))
u,s,vt = np.linalg.svd(A, full_matrices=False)
print(np.linalg.norm(u.T @ A)) ## the goal

## The reduction: if A ~ (J x N), then: 
## 1) multiplication takes O(J^2 N), produces symmetric (J x J) matrix 
## 2) Symmetric eigenvalue takes O(J^3) time 
## => O(J^2 N) when N > J
#%% Eigenvalue 
%%time
r = A.shape[1]
e,q = np.linalg.eigh(A @ A.T) 
ind = np.argsort(-e)
print(np.linalg.norm(q[:,ind[0:r]].T @ A))

## Conclusion: same complexity, but eigenvalue computation relies on BLAS and is parallelizable
## + benefits from vectorization. 
## Also: Phi will be sparse => possibly subcubic complexity! 
## Though: SVD could also be truncated + sparse, and we could use only the first k-vectors since this is the initial guess


# %% simplex projection 
n, d = 2, 2 # n-simplex in d-dimensions
tri = np.random.uniform(size=(n+1,d))*2
p = np.random.uniform(size=(1,d))
# p = np.mean(tri, axis=0).reshape((1,d))
beta = np.zeros((n+1,1))
b = p - tri[[-1],:] 
A = tri[:-1] - np.ones((n-1,1)) @ tri[[-1],:]
x, res, rank, s = np.linalg.lstsq(A.T, b.T, rcond=None)
beta[:n] = x
beta[-1] = 1.0 - np.sum(beta) # np.sum(beta)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(
	np.hstack((tri[:,0],tri[0,0])),
	np.hstack((tri[:,1],tri[0,1])),
	c = "red"
)
ax.scatter(tri[:,0],tri[:,1],c='r')
ax.scatter(p[:,0],p[:,1],c='b')
for i in range(n+1):
	ax.annotate(i, (tri[i,0], tri[i,1])) 

# qb = project_onto_standard_simplex(np.ravel(beta))
qb = projection_simplex_sort(np.ravel(beta), z = 1.0)
q = np.array(qb) @ tri
ax.scatter(q[0],q[1],c='green')

# Should be 0 
print(p - beta.T @ tri)

from shapely.geometry import Point
from shapely.geometry import LineString

point = Point(np.ravel(p)[0], np.ravel(p)[1])
line = LineString([tri[0,:], tri[1,:]])

x = np.array(point.coords[0])

u = np.array(line.coords[0])
v = np.array(line.coords[len(line.coords)-1])

m = v - u
m /= np.linalg.norm(m, 2)
P = u + m*np.dot(x - u, m)
ax.scatter(P[0],P[1],c='orange')

np.linalg.norm(P - p)
np.linalg.norm(q - p)


def projection_simplex_sort(v, z=1):
	n_features = v.shape[0]
	u = np.sort(v)[::-1]
	cssv = np.cumsum(u) - z
	ind = np.arange(n_features) + 1
	cond = u - cssv / ind > 0
	rho = ind[cond][-1]
	theta = cssv[cond][-1] / float(rho)
	w = np.maximum(v - theta, 0)
	return(w)

def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
	func = lambda x: np.sum(np.maximum(v - x, 0)) - z
	lower = np.min(v) - z / len(v)
	upper = np.max(v)

	for it in range(max_iter):
		midpoint = (upper + lower) / 2.0
		value = func(midpoint)

		if abs(value) <= tau:
				break

		if value <= 0:
				upper = midpoint
		else:
				lower = midpoint

	return(np.maximum(v - midpoint, 0))

projection_simplex_sort(np.ravel(beta))
projection_simplex_bisection(np.ravel(beta))