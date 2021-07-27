# %% 2 x 2 SVD
import numpy as np
from tallem import fast_svd
x = np.random.uniform(size=(3,3))

print(np.linalg.svd(x))

y = fast_svd.fast_svd(x)
print(y)

z = fast_svd.lapack_svd(x)
print(z)

# %% 
import numpy as np
from tallem import fast_svd
tall_mat = np.random.uniform(size=(5,2))
fat_mat = np.random.uniform(size=(2,5))
square_mat = np.random.uniform(size=(4,4))
Z = fast_svd.lapack_svd(tall_mat)
print(Z)
print(np.linalg.svd(tall_mat, full_matrices=False))

print("------------------------")

print(np.linalg.svd(fat_mat, full_matrices=False))
print(fast_svd.lapack_svd(fat_mat))

