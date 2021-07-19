# %% Imports
import numpy as np 
import cppimport.import_hook
sys.path.insert(0, "src/tallem")

# %% Random data 
X = np.random.uniform(size=(10, 2))
y = np.random.uniform(size=(10, 1))

# %% Fast dense/sparse matrix multiplication 
# import carma_svd

# %% Test all the SVD implementations 
bn = example.BetaNuclearDense(1000, 3, 3)
bn.three_svd()



# %% 2 x 2 SVD
# Based on: https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
def svd_2x2(x):
	a,b,c,d = x[0,0], x[0,1], x[1,0], x[1,1]
	theta = 0.5*np.arctan2(2*a*c + 2*b*d, a**2 + b**2 - c**2 - d**2)
	phi = 0.5*np.arctan2(2*a*b + 2*c*d, a**2 - b**2 + c**2 - d**2) 
	ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
	u = np.array([[ct, -st],[st, ct]], dtype=np.float32)
	s1 = a**2 + b**2 + c**2 + d**2
	s2 = np.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4*(a*c + b*d)**2)
	s = np.array([ np.sqrt((s1 + s2)/2), np.sqrt((s1 - s2)/2) ])
	sign11 = 1 if ((a*ct+c*st)*cp + (b*ct + d*st)*sp) >= 0 else -1 
	sign22 = 1 if ((a*st-c*ct)*sp + (-b*st + d*ct)*cp) >= 0 else -1 
	v = np.array([[sign11*cp,-sign22*sp],[sign11*sp,sign22*cp]], dtype=np.float32)
	return((u,s,v))



y = np.hstack((x, np.zeros(shape=(3,1))))
print(np.linalg.svd(x))
print(np.linalg.svd(y))

# %% 2 x 2 SVD
import numpy as np
from tallem import fast_svd
x = np.random.uniform(size=(3,3))
y = fast_svd.fast_svd(x)


# %% Try import hook 
import numpy as np
x = np.random.uniform(size=(3,3))
import cppimport.import_hook
sys.path.insert(0, "src/tallem")
import fast_svd

fast_svd.fast_svd(x)

# %% carma 
from tallem import carma_svd
X = np.random.uniform(shape=(10,10))
y = np.random.uniform(shape=(10,1))
print(carma_svd.ols(X, y))