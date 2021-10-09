import numpy as np
from tallem.landmark import landmarks
from tallem.sc import simp_comp, simp_search, delta0, delta0A, delta0D, landmark_cover, eucl_dist

## Get random data + landmarks 
X = np.random.uniform(size=(30,2))
L = landmarks(X, k = 5)
r = np.min(L['radii'])

landmark_pts = X[L['indices'], :]
cover = landmark_cover(X, landmark_pts, r)