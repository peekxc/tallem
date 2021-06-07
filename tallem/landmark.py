# %% Landmark Imports 
import itertools as it
import numpy as np
import numpy.typing as npt
from greedypermutation import clarksongreedy
from greedypermutation.point import Point
from tallem.distance import dist


# %% Landmark Definitions
def find_points(query: npt.ArrayLike, reference: npt.ArrayLike):
	'''
	Given two point clouds 'query' and 'reference', finds the row index of every point in 
	the query set in the reference set if it exists, or -1 otherwise. Essentially performs 
	two binary searches on the first dimension to extracts upper and lower bounds on points 
	with potentially duplicate first dimensions, and then resorts to a linear search on such 
	duplicates
	'''
	Q, R = np.array(query), np.array(reference)
	m = R.shape[0]
	lex_indices = np.lexsort(R.T[tuple(reversed(range(R.shape[1]))),:])
	lb_idx = np.searchsorted(a = R[lex_indices,0], v = Q[:,0])
	ub_idx = np.searchsorted(a = R[lex_indices,0], v = Q[:,0], side="right")
	indices = np.empty(Q.shape[0], dtype = np.int32)
	for i in range(Q.shape[0]):
		if lb_idx[i] >= m:
			indices[i] = -1
		elif lb_idx[i] == ub_idx[i]:
			check_idx = lex_indices[lb_idx[i]]
			indices[i] = check_idx if np.all(Q[i,:] == R[check_idx,:]) else -1
		elif lb_idx[i] == (ub_idx[i] - 1):
			indices[i] = lex_indices[lb_idx[i]]
		else: 
			found = np.where((R[lex_indices[lb_idx[i]:ub_idx[i]],:] == Q[i,:]).all(axis=1))
			indices[i] = -1 if len(found) == 0 else found[0][0]
	return(indices)


def landmarks(a: npt.ArrayLike, k = "default", method = "maxmin"):
	a = np.array(a)
	n = a.shape[0]
	k = n if (k == "default" or k == None) else int(k)
	pc = [Point(a[i,:]) for i in range(n)]
	greedy_lm = [{ "point": p[0], "predecessor": p[1] } for p in it.islice(clarksongreedy._greedy(pc), k)]
	landmarks = np.array([x["point"] for x in greedy_lm])
	landmark_idx = find_points(landmarks, np.array(pc))
	predecessors = np.array([landmark_idx[x["predecessor"]] for x in greedy_lm[1:]], dtype=np.int32)
	cover_radii = dist(a[landmark_idx[1:],:], a[predecessors,:], pairwise = True)
	cover_radii = np.insert(cover_radii, 0, float('inf'), axis = 0)
	return({ "indices" : landmark_idx, "radii" : cover_radii })

# %%
