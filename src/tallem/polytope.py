# polytope.py
# Much of this code is based on code written here: 
# https://github.com/kboulif/Non-negative-matrix-factorization-NMF-/blob/648ea6026a6ef11919f87885fd16ffebf51e5e28/NMF.py

import numpy as np
from typing import *
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import linprog


def intersect_line_plane(n, p, u, x, eps=1.1920929e-07):
	''' Computes the intersections of line with a plane.
	
	Parameters: 
		n := normal vector of the plane
		p := any point in the plane
		u := unit vector giving the direction of the ray/line
		x := any point in the line
		eps := tolerance to detect whether line and plane are parallel (defaults to 32-bit floating point machine tolerance)
	'''
	proj_ray = np.dot(n, u)
	if (abs(proj_ray) <= eps):
		return(np.repeat(np.inf, len(p)))
	w = (x - p)
	d = -np.dot(n, w)/proj_ray
	return(w + d*u + p, d)

def in_hull(x: ArrayLike, hull: Union[ArrayLike, ConvexHull]):
	''' 
	Computes whether a point 'x' lies within the convex hull defined by 'hull'
	
	If 'hull' is an array points, 'x' is just checked to see whether it is expressible as a convex combination of the points in 'hull'
	
	Otherwise, if 'hull' is a pre-computed convex hull (from scipy.spatial), then it is checked whether 'x' lies in the intersection 
	of every halfspace defined by the H-representation of the 'hull.' 
	'''
	if isinstance(hull, ConvexHull):
		eps = np.finfo(np.float32).eps
		return(np.all(np.dot(x, hull.equations[:,:-1].T) + hull.equations[:,-1] <= eps))
	else: 
		assert isinstance(hull, np.ndarray) and hull.ndim == 2
		n, d = hull.shape
		c = np.zeros(n)
		A = np.r_[hull.T,np.ones((1,n))]
		b = np.r_[x, np.ones(1)]
		lp = linprog(c, A_eq=A, b_eq=b)
		return lp.success

def project_line(X, x0, bary=False):
	''' Projects point x0 onto line segment X=(x1, x2) where X == (d x 2) matrix defining the line segment'''
	x1, x2 = X[:,0], X[:,1]
	alpha = float(np.dot(np.transpose(x1-x2), x0-x2))/(np.dot(np.transpose(x1-x2), x1-x2))
	alpha = max(0,min(1,alpha))
	y = alpha*x1 + (1-alpha)*x2
	theta = np.array([alpha, 1-alpha])
	return((theta, y) if bary else y)

def project_triangle(X, x0, bary=False):
	''' Projects point x0 onto a triangle X=(x1, x2, x3) where X == (d x 3) matrix defining the triangle'''
	d = len(x0)
	XX = np.zeros((d,2))
	XX[:,0] = X[:,0] - X[:,2]
	XX[:,1] = X[:,1] - X[:,2] 
	P = np.dot(np.linalg.inv(np.dot(np.transpose(XX),XX)),np.transpose(XX))
	theta = np.append(np.dot(P, x0-X[:,2]), 1-np.sum(np.dot(P, x0-X[:,2])))
	y = np.dot(X,theta)
	if ((any(theta<0)) or (any(theta>1)) or (np.sum(theta)!=1)):
		d1, d2, d3 = np.linalg.norm(X.T - y, axis = 1)
		theta4,y4 = project_line(X[:,[0,1]],y,bary=True) 
		theta5,y5 = project_line(X[:,[0,2]],y,bary=True)
		theta6,y6 = project_line(X[:,[1,2]],y,bary=True)
		d4,d5,d6 = np.linalg.norm(y-y4),np.linalg.norm(y-y5),np.linalg.norm(y-y6)
		d = min(d1,d2,d3,d4,d5,d6)
		if (d1 == d):
			y = X[:,0]
			theta = np.array([1,0,0])
		elif (d2 == d):
			y = X[:,1]
			theta = np.array([0,1,0])
		elif (d3 == d):
			y = X[:,2]
			theta = np.array([0,0,1])
		elif (d4 == d):
			y = y4
			theta = np.zeros(3)
			theta[[0,1]] = theta4
		elif (d5 == d):
			y = y5
			theta = np.zeros(3)
			theta[[0,2]] = theta5
		else:
			y = y6
			theta = np.zeros(3)
			theta[[1,2]] = theta6
	return((theta, y) if bary else y)

def project_tetrahedron(X, x0, bary=False):
	'''
	Projects point x0 onto a tetrahedron X=(x1, x2, x3) where X == (d x 4) matrix defining the tetrahedron
	'''
	d = len(x0)
	XX = np.zeros((d,3))
	XX[:,0] = X[:,0] - X[:,3]
	XX[:,1] = X[:,1] - X[:,3]
	XX[:,2] = X[:,2] - X[:,3]
	P = np.dot(np.linalg.inv(np.dot(np.transpose(XX),XX)),np.transpose(XX))
	theta = np.append(np.dot(P, x0-X[:,3]), 1-np.sum(np.dot(P, x0-X[:,3])))
	y = np.dot(X,theta)
	if ((any(theta<0)) or (any(theta>1)) or (np.sum(theta)!=1)):
		d1,d2,d3,d4 = np.linalg.norm(X.T - x0, axis=1)
		d5,d6,d7,d8,d9 = np.linalg.norm(y-y5),np.linalg.norm(y-y6),np.linalg.norm(y-y7), np.linalg.norm(y-y8),np.linalg.norm(y-y9)
		d10,d11,d12,d13,d14 = np.linalg.norm(y-y10),np.linalg.norm(y-y11),np.linalg.norm(y-y12),np.linalg.norm(y-y13),np.linalg.norm(y-y14)
		theta5,y5 = project_line(X[:,[0,1]],y,bary=True) 
		theta6,y6 = project_line(X[:,[0,2]],y,bary=True)
		theta7,y7 = project_line(X[:,[0,3]],y,bary=True)
		theta8,y8 = project_line(X[:,[1,2]],y,bary=True) 
		theta9,y9 = project_line(X[:,[1,3]],y,bary=True)
		theta10,y10 = project_line(X[:,[2,3]],y,bary=True) 
		theta11,y11 = project_triangle(X[:,[0,1,2]],y,bary=True)
		theta12,y12 = project_triangle(X[:,[0,1,3]],y,bary=True)
		theta13,y13 = project_triangle(X[:,[0,2,3]],y,bary=True)
		theta14,y14 = project_triangle(X[:,[1,2,3]],y,bary=True)
		d = min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14)
		if (d1 == d):
			y = X[:,0]
			theta = np.array([1,0,0,0])
		elif (d2 == d):
			y = X[:,1]
			theta = np.array([0,1,0,0])
		elif (d3 == d):
			y = X[:,2]
			theta = np.array([0,0,1,0])
		elif (d4 == d):
			y = X[:,3]
			theta = np.array([0,0,0,1])
		elif (d5 == d):
			y = y5
			theta = np.zeros(4)
			theta[[0,1]] = theta5
		elif (d6 == d):
			y = y6
			theta = np.zeros(4)
			theta[[0,2]] = theta6
		elif (d7 == d):
			y = y7
			theta = np.zeros(4)
			theta[[0,3]] = theta7
		elif (d8 == d):
			y = y8
			theta = np.zeros(4)
			theta[[1,2]] = theta8
		elif (d9 == d):
			y = y9
			theta = np.zeros(4)
			theta[[1,3]] = theta9
		elif (d10 == d):
			y = y10
			theta = np.zeros(4)
			theta[[2,3]] = theta10
		elif (d11 == d):
			y = y11
			theta = np.zeros(4)
			theta[[0,1,2]] = theta11
		elif (d12 == d):
			y = y12
			theta = np.zeros(4)
			theta[[0,1,3]] = theta12
		elif (d13 == d):
			y = y13
			theta = np.zeros(4)
			theta[[0,2,3]] = theta13
		else:
			y = y14
			theta = np.zeros(4)
			theta[[1,2,3]] = theta14
	return((theta, y) if bary else y)

# From: https://github.com/kboulif/Non-negative-matrix-factorization-NMF-/blob/648ea6026a6ef11919f87885fd16ffebf51e5e28/NMF.py
def project_wolfe(X,epsilon=1e-6,threshold=1e-8,niter=100, include_weights=False, verbose=False):
	''' 
	Projects origin onto the convex hull of the rows of 'X' using Wolfe method. The algorithm is by Wolfe in his paper 'Finding the nearest point in A polytope'. 
	Parameters:	
		'epsilon', 'threshold': Algorithm parameters determining approximation acceptance thresholds. These parameters are denoted as (Z2,Z3) and Z1, in the main paper, respectively. Default values = 1e-6, 1e-8.
		'niter': Maximum number of iterations. Default = 10000.	
		'verbose': If set to be True, the algorithm prints out the current set of weights, active set, current estimate of the projection after each iteration. Default = False.
	'''
	assert isinstance(X, np.ndarray) and X.ndim == 2, "'X' must be a 2-dimensional numpy matrix"
	n, d = X.shape
	max_norms = np.min(np.sum(np.abs(X)**2,axis=-1)**(1./2))
	s_ind = np.array([np.argmin(np.sum(np.abs(X)**2,axis=-1)**(1./2))])
	w = np.array([1.0])
	E = np.array([[-max_norms**2, 1.0], [1.0, 0.0]])
	isoptimal = 0
	c = 0
	while (isoptimal == 0) and (c <= niter):
		c, isoptimal_aff = c+1, 0
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
		if (verbose): print(f'X_s={X[s_ind,:]}, w={w}, y={y},s_ind={s_ind}')
		weights = np.zeros(n)
		weights[s_ind] = w
	return((y, weights) if include_weights else y)


def project_hull_fast(p: ArrayLike, hull_vertices: ArrayLike):
	''' 
	Projects a point 'p' onto the convex hull defined by 'hull.'

	If 'p' is on the interior of 'hull', then the projection of the point is the point itself.
	'''
	d = len(p)
	if d == 1 and isinstance(hull, ArrayLike) and hull.shape[1] == 1:
		lb, ub = np.min(hull), np.max(hull)
		if p <= lb: return(lb)
		elif p >= ub: return(ub)
		return(p)
	facet_dim = hull_vertices.shape[1]
	if facet_dim > 4: 
		z = project_wolfe(hull_vertices-p)+p
	else:
		ind = np.flatnonzero(np.array([2,3,4]) == facet_dim)[0]
		proj_f = ([project_line, project_triangle, project_tetrahedron])[ind]
		z = proj_f(hull_vertices.T, p)
	return(z)
	# facet_dim = len(hull.vertices)

def project_hull(X: ArrayLike, hull: Union[ArrayLike, ConvexHull], method: str = "interior"):
	''' 
	Projects points 'X' onto parts of the convex hull 'hull'

	The convex hull of 
	'''
	if not(isinstance(hull, ConvexHull)):
		hull = ConvexHull(hull)
	assert isinstance(hull, ConvexHull)
	hull_vertices = hull.points[hull.vertices,:]
	if method == "interior":
		Z = np.array([project_hull_fast(x, hull_vertices) for x in X])
	elif method == "complement":
		## TODO: check for interior
		hull_simplices = hull.points[hull.simplices]
		Z = []
		for x in X: 
			P = np.array([project_hull_fast(x, facet) for facet in hull_simplices])
			z = P[np.argmin(np.linalg.norm(P - p, axis = 1)),:]
			Z.append(z)
		Z = np.array(Z)
	elif method == "boundary": 
		hull_simplices = hull.points[hull.simplices]
		Z = []
		for x in X: 
			P = np.array([project_hull_fast(x, facet) for facet in hull_simplices])
			z = P[np.argmin(np.linalg.norm(P - p, axis = 1)),:]
			Z.append(z)
		Z = np.array(Z)

	facet_dim = len(hull.vertices)
	hull_vertices = hull.points[hull.vertices,:]

def sdist_to_boundary(X: ArrayLike, hull: Union[ArrayLike, ConvexHull], project_facet: Callable, coords: bool = False):
	'''
		Computes the signed distance to the boundary of an arbitrary polytope.
		
		For each point x \in X, if x lies on the interior of the convex hull defined by 'hull', 
		then the reported distance for x is the L2 norm between x and its minimum distance
		projection onto the boundary of 'hull'. 

		If x lies outside the hull, the distances are reported as negative.

		Note that, in either case, the distance between x and the convex hull is 0 if and only if 
		x lies on the boundary of the hull. 
		
		Parameters: 
			X := (n x d) matrix of points in R^d
			hull := (k x d) matrix of points in R^d, or an instance of scipy.spatial.qhull.ConvexHull 
		
		Return: 
			min_dist := numpy array with size (n,) giving the (signed) distances between the points 'x' and 'hull'.
			points := if coords=True, then the projected points are also returned.
	'''
	if isinstance(hull, ConvexHull):
		delh = Delaunay(hull.points)
	else: 
		hull = ConvexHull(hull)
		delh = Delaunay(hull.points)
	
	hull_vertices = np.vstack((hull.points[hull.vertices,:], hull.points[hull.vertices[0]]))
	n = hull_vertices.shape[0]
	min_dist = np.full(X.shape[0], np.inf)
	projected_pts = np.zeros(X.shape)
	for i in range(n-1):
		Z = np.array([project_facet(hull_vertices[i:(i+2),:].T, x) for x in X])
		proj_dist = np.array([np.linalg.norm(x - z) for x,z in zip(X, Z)])
		min_dist = np.minimum(proj_dist, min_dist)
		to_replace = np.flatnonzero(proj_dist == min_dist)
		projected_pts[to_replace,:] = Z[to_replace,:]
	
	## Mark distances with the correct sign depending if they are outside or inside the polytope
	is_outside = np.flatnonzero(delh.find_simplex(X) < 0)
	if len(is_outside) > 0:
		min_dist[is_outside] = -min_dist[is_outside]
	return(min_dist, projected_pts)
