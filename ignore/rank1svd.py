import os
import numpy as np 
from src.tallem import fast_svd
np.random.seed(0)
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.8g" % x))
#A = np.random.uniform(size=(4,5))
#U,S,Vt = np.linalg.svd(A, full_matrices=False)
#a, b = np.random.uniform(size=(4,1))*0.05, np.random.uniform(size=(5,1))*0.05

A = np.random.uniform(size=(4,4))
A = A.T @ A
d, Q = np.linalg.eigh(A)
ind = np.argsort(d)
d, Q = d[ind], Q[:,ind]
u = np.random.uniform(size=(4,1))
u /= np.linalg.norm(u)
# u = np.repeat(1.0, A.shape[0])

from src.tallem import fast_svd
fast_svd.dpr1_ev(Q, d, 0.5, u) ## rank-1 update to eigenvalues + eigenvectors 


## Yields DPR1
L = np.linalg.eig(np.diag(d) + 0.05*u@u.T)[0]
fast_svd.dpr1(d, 0.05, u, 1)

np.linalg.eig(np.diag(d) + 0.05*u@u.T)[1]

EV = [((1.0/(d - np.repeat(l, len(d)))) * u.T).T for l in L]
np.hstack([-(E / np.linalg.norm(E)) for E in EV])
	
## 
np.linalg.eigh(Q@(np.diag(d) + 0.5*u@u.T)@Q.T)

# %%
import numpy as np 
np.random.seed(0)
A = np.random.uniform(size=(4,4))


## Truth 
Us, Ss, Vts = np.linalg.svd(A + a @ b.T)

## naive method?



# %% Update method for U 
U,s,Vt = np.linalg.svd(A, full_matrices=False)
a, b = np.random.uniform(size=(A.shape[0],1))*0.05, np.random.uniform(size=(A.shape[1],1))*0.05
alpha, beta, a_tilde, b_tilde = a.T @ a, b.T @ b, Vt.T @ np.diag(s) @ U.T @ a, A @ b
d = s**2

## Left singular vectors
rho, Q = np.linalg.eigh(np.array([[beta.item(), 1], [1, 0]]))
a_bar, b_bar = np.hsplit(np.hstack((a, b_tilde)) @ Q, [0,1])[1:]

## There has to be something wrong with DPR1 code. Need to recheck test cases with basic EV update 
# fast_svd.dpr1(d, -rho[0], a_bar/np.linalg.norm(a_bar), 3)
# np.linalg.eigh(np.diag(d) + rho[0]*(a_bar @ a_bar.T))
# fast_svd.dpr1_ev(U, d, rho[0], a_bar) 

ct, ut = dpr1_update(U, d, rho[0], a_bar)
ct, ut = dpr1_update(ut, ct, rho[1], b_bar)

## Right singular vectors
V = Vt.T
phi, R = np.linalg.eigh(np.array([[alpha.item(), 1], [1, 0]]))
b_ubar, a_ubar = np.hsplit(np.hstack((b, a_tilde)) @ R, [0,1])[1:]
ct, vt = dpr1_update(V, s, phi[0], b_ubar)
ct, vt = dpr1_update(vt, ct, phi[1], a_ubar)

ut @ np.diag(np.sqrt(ct)) @ vt.T

# vt = np.diag(s) @ ut.T @ (A + a @ b.T) # updated right singular vectors (nope)

# %% 
U @ np.diag(S) @ Vt @ b

## THis is also true! 
(Us @ np.diag(Ss) @ np.diag(Ss).T @ Us.T)
U @ np.diag(S) @ np.diag(S).T @ U.T + b_tilde @ a.T + a @ b_tilde.T + beta * a @ a.T

## 2x2 reduction
ab = np.hstack((a, b_tilde))
B = np.array([[beta.item(), 1], [1, 0]])

## This is true!
t1 = (U @ np.diag(S) @ np.diag(S).T @ U.T) + (ab @ B @ ab.T)
t2 = (Us @ np.diag(Ss) @ np.diag(Ss).T @ Us.T)

## Diagonalize Q-factor
rho, Q = np.linalg.eigh(B)
tmp = ab @ Q
a_bar, b_bar = tmp[:,[0]], tmp[:,[1]]

## Check Q equations (not even close!)
#np.linalg.eigh(ab @ Q @ np.diag(rho) @ Q.T @ ab.T)
# np.linalg.eigh((A + a@b.T)@(A + a@b.T).T)
# np.linalg.eigh((U @ np.diag(S) @ np.diag(S).T @ U.T) + (ab @ Q @ np.diag(rho) @ Q.T @ ab.T))

## Update w/ two symmetric rank-1 updates
a_hat = U.T @ a_bar
u = a_hat / np.linalg.norm(a_hat)
rho_norm = np.linalg.norm(a_hat)**2
fast_svd.dpr1_ev(U, d, rho_norm, u) ## should equal first update


# need: U @ D @ U.T + rho[0]*rho_norm*u@u.T
u = a_hat / np.linalg.norm(a_hat)
rho_norm = np.linalg.norm(rho[0]*(a_bar @ a_bar.T))
fast_svd.dpr1_ev(U, d, rho_norm, u) 

evu = np.sort(fast_svd.dpr1_ev(U, d, rho_norm, u)['eval'].T)
evp = np.sort(np.linalg.eig(D + rho[0]*a_hat@a_hat.T)[0])
evm = np.sort(np.linalg.eig(D - rho[0]*a_hat@a_hat.T)[0])

evu + 2*(np.diag(D) - evu)

## Non-normalized
D = np.diag(d)
np.linalg.eig(U @ D @ U.T + rho[0]*a_bar@a_bar.T) ## what is wanted

def dpr1_update(U, d, rho, a):
	''' 
	Performs a diagonal plus symmetric rank-1 updates to an eigendecomposition

	Given an eigen decomposition A = U D U^T fo some matrix A, this function 
	updates A + rho x (a @ a^T) = U D U^T + rho * (a @ a^T)
	
	(U @ diag(d) @ U^T) + rho * (a @ a^T) =  U @ (diag(d) + rho * u @ u^T) @ U^T where u = U^T a

	'''
	assert np.prod(d.shape) == len(d) and d.ndim == 1, "d must be one-dimensional"
	# assert rho > 0, "rho must be a positive scalar"
	u = a / np.linalg.norm(a)
	rho_norm = np.linalg.norm(rho*(a @ a.T))
	res = fast_svd.dpr1_ev(U, d, rho_norm, u) 
	if res['info'] != 0: raise ValueError("Update failed with status {}".format(res['info']))
	res['eval'] = np.ravel(res['eval'])
	if rho < 0: res['eval'] = res['eval'] + 2*(d - res['eval'])
	return(res['eval'], res['evec'])

ct, ut = dpr1_update(U, d, rho[0], a_bar)
ct, ut = dpr1_update(ut, ct, rho[1], b_bar)
		
V = np.diag(ct) @ ut.T @ (A + a @ b.T)

np.linalg.eig(U @ D @ U.T + rho[0]*a_bar@a_bar.T + rho[1]*b_bar@b_bar.T)[0]

## Do it! 
D = np.diag(S) @ np.diag(S).T 
U @ D @ U.T ## Symmetric and PD, but not orthonormal! 
a_hat = U.T @ a_bar

## The truth 
np.linalg.eig(D + rho[0]*a_hat@a_hat.T)

## (doesn't work!) From http://www.stat.uchicago.edu/~lekheng/courses/309f10/modified.pdf
## DPR1 -> tridiagonal 
# from scipy.linalg import eigh_tridiagonal
# ind = np.argsort(np.ravel(np.abs(a_hat)))
# u = (np.ravel(a_hat)[ind]).reshape((len(ind), 1))
# r = np.array([n/d if d != 0 else 0 for n,d in zip(-u[:-1], u[1:])])
# assert np.all(np.abs(r) <= 1)
# K = np.eye(len(u)) + np.diag(r, 1)
# Kt = (K @ D @ K.T) + rho[0] * (K @ (u @ u.T) @ K.T)
# # eigh(Kt, K@K.T) ## generalized eigenvalues should match truth above
# D_tilde, C_tilde = eigh_tridiagonal(np.diag(Kt), np.diag(Kt, 1))

## The result 
# U_tilde = U @ C_tilde
# rind = np.argsort(ind)
# res = U_tilde[:,rind], D_tilde[rind]

## secular equation
# from scipy.optimize import root_scalar
# omega = lambda e: 1 + rho[0]*np.sum((np.ravel(u)**2)/(np.diagonal(D) - e))
# root_scalar(omega)

# dir(np.linalg.lapack_lite)
# from scipy.linalg import lapack 
# dir(lapack)

#ind = np.argpartition(np.abs(rho), -2)[-2:]
#rho, Q = rho[ind], Q[ind,ind]

D_tilde, C_tilde = np.linalg.eigh(D + rho[0]*a_hat@a_hat.T)
U_tilde = U @ C_tilde
b_hat = U_tilde @ b_bar
np.linalg.eigh(D_tilde + rho[1]*b_hat@b_hat.T)

## Nothing works
np.sqrt((np.linalg.eigh(D_tilde + rho[1]*b_hat@b_hat.T))[0])

## Verify sum of outer products works (this works!)
np.linalg.eigh((A + a@b.T) @ (A + a@b.T).T)[0]
np.linalg.eig((U @ D @ U.T) + rho[0] * a_bar @ a_bar.T + rho[1] * b_bar @ b_bar.T)

def dpr1(d, rho, u):
	return(np.linalg.eigh(d + rho*u@u.T))

## Verify a_hat relation to a_bar (works)
dpr1(D, rho[0], a_hat)[0]
np.linalg.eigh(U @ D @ U.T + rho[0]*a_bar@a_bar.T)[0]

# now RESET: try b! 
dt, ut = np.linalg.eigh(U @ D @ U.T + rho[0]*a_bar@a_bar.T)
D_tilde = np.diag(dt)

dpr1(D_tilde, rho[1], b_bar)[0]
np.linalg.eigh(ut @ D_tilde @ ut.T + rho[1]*b_bar@b_bar.T)[0]


ub, cb = dpr1(U @ D @ U.T, rho[0], a_bar)


dpr1(ub, rho[1], (U @ cb).T @ b_bar)

## Verify singular vectors (works up to sign difference)
D_tilde, C_tilde = dpr1(D, rho[0], a_hat)
(U @ C_tilde) - dpr1(U @ D @ U.T, rho[0], a_bar)[1]
U_tilde = (U @ C_tilde)

### Verify double add (doesn't work!) b_bar can't be right
dpr1(U_tilde@D_tilde@U_tilde.T, rho[1], b_bar)
np.linalg.eig((U @ D @ U.T) + rho[0] * a_bar @ a_bar.T + rho[1] * b_bar @ b_bar.T)


## First ev update works, second doesn't!
true_s = np.linalg.eig((U @ D @ U.T) + rho[0] * a_bar @ a_bar.T)[0]
dt, ct = dpr1(D, rho[0], a_hat)
np.ravel(np.sort(true_s)) - np.ravel(np.sort(dt))

## Try singular vectors on b_tilde
u_tilde = np.linalg.eig((U @ D @ U.T) + rho[0] * a_bar @ a_bar.T)[1]
dpr1(np.diag(dt), rho[1], U.T @ b_tilde)[0]




true_s = np.linalg.eig((U @ D @ U.T) + rho[0] * a_bar @ a_bar.T + rho[1] * b_bar @ b_bar.T)[0]

dt, ut = dpr1(D, rho[0], a_hat)
dpr1(np.diag(dt), rho[1], ut @ b_tilde)[0]
dpr1(np.diag(dt), rho[1], (U @ C_tilde) @ b_tilde)[0]


# %% DPR1 
def dpr1_valid(d, rho, u):
	rho * (u @ u.T)



