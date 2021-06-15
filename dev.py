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

