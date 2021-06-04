# %% Landmark Imports 
import itertools as it
from greedypermutation import clarksongreedy
from greedypermutation.point import Point

# %% Landmark Definitions
def find_points(query, reference):
	'''
	Given two point clouds 'query' and 'reference', finds the row index of every point in 
	the query set in the reference set if it exists, or -1 otherwise. Essentially performs 
	two binary searches on the first dimension to extracts upper and lower bounds on points 
	with potentially duplicate first dimensions, and then resorts to a linear search on such 
	duplicates
	'''
	Q, R = query, reference
	lex_indices = np.lexsort(R.T[tuple(reversed(range(R.shape[1]))),:])
	lb_idx = np.searchsorted(a = R[lex_indices,0], v = Q[:,0])
	ub_idx = np.searchsorted(a = R[lex_indices,0], v = Q[:,0], side="right")
	indices = np.empty(Q.shape[0], dtype = np.int32)
	for i in range(Q.shape[0]):
		indices[i] = -1
		for idx in range(lb_idx[i], ub_idx[i]):
			if all(Q[i,:] == R[lex_indices[idx],:]):
				indices[i] = lex_indices[idx]
	return(indices)


def landmarks(X, k = X.shape[0], method = "maxmin"):
	n = X.shape[0]
	pc = [Point(X[i,:]) for i in range(n)]
	greedy_lm = [{ "point": p[0], "predecessor": p[1] } for p in it.islice(clarksongreedy._greedy(pc), k)]
	landmarks = np.array([x["point"] for x in greedy_lm])
	landmark_idx = find_points(landmarks, np.array(pc))
	predecessors = np.array([x["predecessor"] for x in greedy_lm[1:]], dtype=np.int32)
	cover_radii = dist(x=X[landmark_idx[1:],:], y=X[predecessors,:], pairwise = True)
	cover_radii = np.insert(cover_radii, 0, float('inf'), axis = 0)
	return({ "indices" : landmark_idx, "radii" : cover_radii })

