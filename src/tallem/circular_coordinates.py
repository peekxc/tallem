## Much of this is taken verbatim from: https://github.com/ctralie/DREiMac/blob/master/dreimac/circularcoords.py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from ripser import ripser
import warnings

def reindex_cocycles(cocycles, idx_land, N):
  """
  Convert the indices of a set of cocycles to be relative
  to a list of indices in a greedy permutation
  Parameters
  ----------
  cocycles: list of list of ndarray
      The cocycles
  idx_land: ndarray(M, dtype=int)
      Indices of the landmarks in the greedy permutation, with
      respect to all points
  N: int
      Number of total points
  """
  idx_map = -1*np.ones(N, dtype=int)
  idx_map[idx_land] = np.arange(idx_land.size)
  for ck in cocycles:
    for c in ck:
      c[:, 0:-1] = idx_map[c[:, 0:-1]]
          
def add_cocycles(c1, c2, p = 2, real = False):
  S = {}
  c = np.concatenate((c1, c2), 0)
  for k in range(c.shape[0]):
    [i, j, v] = c[k, :]
    i, j = min(i, j), max(i, j)
    if not (i, j) in S:
      S[(i, j)] = v
    else:
      S[(i, j)] += v
  cret = np.zeros((len(S), 3))
  cret[:, 0:2] = np.array([s for s in S])
  cret[:, 2] = np.array([np.mod(S[s], p) for s in S])
  dtype = np.int64
  if real:
    dtype = np.float32
  cret = np.array(cret[cret[:, -1] > 0, :], dtype = dtype)
  return cret


def make_delta0(R):
  """
  Return the delta0 coboundary matrix
  :param R: NEdges x 2 matrix specifying edges, where orientation
  is taken from the first column to the second column
  R specifies the "natural orientation" of the edges, with the
  understanding that the ranking will be specified later
  It is assumed that there is at least one edge incident
  on every vertex
  """
  NVertices = int(np.max(R) + 1)
  NEdges = R.shape[0]
  
  #Two entries per edge
  I = np.zeros((NEdges, 2))
  I[:, 0] = np.arange(NEdges)
  I[:, 1] = np.arange(NEdges)
  I = I.flatten()
  J = R[:, 0:2].flatten()
  V = np.zeros((NEdges, 2))
  V[:, 0] = -1
  V[:, 1] = 1
  V = V.flatten()
  I = np.array(I, dtype=int)
  J = np.array(J, dtype=int)
  Delta = coo_matrix((V, (I, J)), shape=(NEdges, NVertices)).tocsr()
  return Delta

class CircularCoords():
  def __init__(self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False):
    """
    Parameters
    ----------
    X: ndarray(N, d)
        A point cloud with N points in d dimensions
    n_landmarks: int
        Number of landmarks to use
    distance_matrix: boolean
        If true, treat X as a distance matrix instead of a point cloud
    prime : int
        Field coefficient with which to compute rips on landmarks
    maxdim : int
        Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
        but it may be of interest to see other dimensions (e.g. for a torus)
    """
    assert(maxdim >= 1)
    self.verbose = verbose
    res = ripser(X, distance_matrix=distance_matrix, coeff=prime, maxdim=maxdim, n_perm=n_landmarks, do_cocycles=True)
    self.X_ = X
    self.prime_ = prime
    self.dgms_ = res['dgms']
    self.dist_land_data_ = res['dperm2all']
    self.idx_land_ = res['idx_perm']
    self.dist_land_land_ = self.dist_land_data_[:, self.idx_land_]
    self.cocycles_ = res['cocycles']

    # Sort persistence diagrams in descending order of persistence
    idxs = np.argsort(self.dgms_[1][:, 0]-self.dgms_[1][:, 1])
    self.dgms_[1] = self.dgms_[1][idxs, :]
    self.dgm1_lifetime = np.array(self.dgms_[1])
    self.dgm1_lifetime[:, 1] -= self.dgm1_lifetime[:, 0]
    self.cocycles_[1] = [self.cocycles_[1][idx] for idx in idxs]
    reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])
    self.n_landmarks_ = n_landmarks
    self.type_ = "emcoords"
    self.type_ = "circ"

  def get_coordinates(self, perc = 0.99, do_weighted = False, cocycle_idx = [0], pou = "identity"):
    """
    Perform circular coordinates via persistent cohomology of 
    sparse filtrations (Jose Perea 2018)
    Parameters
    ----------
    perc : float
        Percent coverage
    do_weighted : boolean
        Whether to make a weighted cocycle on the representatives
    cocycle_idx : list
        Add the cocycles together in this list
    pou: (dist_land_data, r_cover) -> phi
        A function from the distances of each landmark to a bump function
    """
    ## Step 1: Come up with the representative cocycle as a formal sum
    ## of the chosen cocycles
    n_landmarks = self.n_landmarks_
    n_data = self.X_.shape[0]
    dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
    cohomdeath = -np.inf
    cohombirth = np.inf
    cocycle = np.zeros((0, 3))
    prime = self.prime_
    for k in range(len(cocycle_idx)):
      cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=prime)
      cohomdeath = max(cohomdeath, dgm1[cocycle_idx[k], 0])
      cohombirth = min(cohombirth, dgm1[cocycle_idx[k], 1])

    ## Step 2: Determine radius for balls
    dist_land_data = self.dist_land_data_
    dist_land_land = self.dist_land_land_
    coverage = np.max(np.min(dist_land_data, 1))
    r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
    self.r_cover_ = r_cover # Store covering radius for reference
    
    ## Step 3: Setup coboundary matrix, delta_0, for Cech_{r_cover }
    ## and use it to find a projection of the cocycle
    ## onto the image of delta0

    #Lift to integer cocycle
    val = np.array(cocycle[:, 2])
    val[val > (prime-1)/2] -= prime
    Y = np.zeros((n_landmarks, n_landmarks))
    Y[cocycle[:, 0], cocycle[:, 1]] = val
    Y = Y + Y.T
    #Select edges that are under the threshold
    [I, J] = np.meshgrid(np.arange(n_landmarks), np.arange(n_landmarks))
    I = I[np.triu_indices(n_landmarks, 1)]
    J = J[np.triu_indices(n_landmarks, 1)]
    Y = Y[np.triu_indices(n_landmarks, 1)]
    idx = np.arange(len(I))
    idx = idx[dist_land_land[I, J] < 2*r_cover]
    I = I[idx]
    J = J[idx]
    Y = Y[idx]

    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I
    #Make a flat array of NEdges weights parallel to the rows of R
    if do_weighted:
      W = dist_land_land[I, J]
    else:
      W = np.ones(NEdges)
    delta0 = make_delta0(R)
    wSqrt = np.sqrt(W).flatten()
    WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
    A = WSqrt*delta0
    b = WSqrt.dot(Y)
    tau = lsqr(A, b)[0]
    theta = np.zeros((NEdges, 3))
    theta[:, 0] = J
    theta[:, 1] = I
    theta[:, 2] = -delta0.dot(tau)
    theta = add_cocycles(cocycle, theta, real=True)
    
    ## Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
    U = dist_land_data < r_cover
    phi = np.zeros_like(dist_land_data)
    
    ## linear 
    if pou == "identity" or pou == "linear":
      phi[U] = r_cover - dist_land_data[U]
    elif pou == "quadratic":
      phi[U] = r_cover - dist_land_data[U]
    elif pou == "gaussian":
      phi[U] = np.exp(r_cover**2/(dist_land_data[U]**2-r_cover**2))
    else: 
      raise ValueError("wrong pou")

    # Compute the partition of unity 
    # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
    denom = np.sum(phi, 0)
    nzero = np.sum(denom == 0)
    if nzero > 0:
      warnings.warn("There are %i point not covered by a landmark"%nzero)
      denom[denom == 0] = 1
    varphi = phi / denom[None, :]

    # To each data point, associate the index of the first open set it belongs to
    ball_indx = np.argmax(U, 0)

    ## Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
    
    # compute all transition functions
    theta_matrix = np.zeros((n_landmarks, n_landmarks))
    I = np.array(theta[:, 0], dtype = np.int64)
    J = np.array(theta[:, 1], dtype = np.int64)
    theta = theta[:, 2]
    theta = np.mod(theta + 0.5, 1) - 0.5
    theta_matrix[I, J] = theta
    theta_matrix[J, I] = -theta
    class_map = -tau[ball_indx]
    for i in range(n_data):
      class_map[i] += theta_matrix[ball_indx[i], :].dot(varphi[:, i])    
    thetas = np.mod(2*np.pi*class_map, 2*np.pi)
    return thetas