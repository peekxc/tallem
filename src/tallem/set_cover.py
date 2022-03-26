import numpy as np
from scipy.sparse import issparse
from scipy.optimize import linprog

# Let U = { u_1, u_2, ..., u_n } denote the universe/set of elements to cover
# Let S = { s_1, s_2, ..., s_m } denote the set of subsets with weights w = { w_1, w_2, ..., w_m }
# Let y = { y_1, y_2, ..., y_m } denote the binary variables s.t. y_i = 1 <=> s_i included in solution 
# The goal is to find the assignment Y producing a cover S* of minimal weight 
# If k > 0 is specified and one wants |S*| = k, then this is the decision variant of the weighted set cover problem
# The ILP formulation: 
# 	minimize sum_{s_j in S} w_j*s_j
# 	subject to
# 		sum_{x in S, s_j in S} y_j >= 1 for each u \in U
# 		y_j \in { 0, 1 } 
# 
# The ILP can be relaxed to LP via letting y_j vary \in [0, 1].
# Let A \in R^{n x m} where A(i,j) = 1 if u_i \in s_j and b = { 1, 1, ..., 1 } with |b| = m. 
# The corresponding LP is: 
# 	minimize 			 w^T * y
# 	subject to 		 Ay >= b 
# 								 0 <= y_j <= 1 		for all j in [m]
#
# From here, we have two strategies (http://theory.stanford.edu/~trevisan/cs261/lecture08.pdf): 
#   1. Rounding approach: If we know each u belongs to at most *k* sets, then we may choose
#  		 y_j = 1 if y_j >= 1/k and y_j = 0 otherwise. This guarentees a feasible cover which is 
# 		 a k-approximation of the objective y*.  
#   2. Randomized approach: Interpret each y_j as probability of including subset s_j, sample from Y*
#			 using y_j as probabilities. Due to LP, expected cost of resulting assignment is upper bounded 
#      the weight of cost(Y*). 
# Note that (1) can only be used if one knows each u belongs to at most *k* sets. However, even in this case 
# if this is known, k may be large, yielding a poor approximation (k = n/2 => ) (see https://www.cs.cmu.edu/~anupamg/adv-approx/lecture2.pdf). 
# Suppose we solve the LP once, obtaining y. Interpreting each y \in [0,1], we let C = {} and repeat:
#   - While C does not cover U:
# 	- 	For each j in [m]:
# 	- 		Assign y_j = 1 w/ probability y_j
# Assume n >= 2. Then:
# 	a. P(u_i not covered) after k iterations = e^{-k} => take e.g. k = c lg n, then P(u_i not covered) <= 1/(n^c)
# 	b. P(there exists u_i not covered) <= \sum\limits_{i=0}^n P(u_i no covered) = n*(1/n^c) = n^(1-c). 
# Thus, for k = c lg(n). One can show that the randomized approach: 
# 	=> produces a feasible cover after k iterations with probability >= 1/(1 - n^(1-c))
# 	=> produces an assignment \hat{y}* with expected cost c*ln(n)*opt(y*)
# In particular, if c = 2 then then with probability p = 1 - 1/n, we will have a 2*ln(n)-approximation after 2*lg(n) iterations.
# At best, we have a \Theta(ln n)-approximation using the LP rounding. 
# (Ideally we wanted (1-\eps)ln(n) approximation! )
def weighted_set_cover(C, W):
	''' 
	Attempts to computes an approximate solution to the weighted set cover problem 
	
	Parameters: 
		C := (n x J) sparse matrix giving the set membership of n points on J subsets 
		W := (J,) array of weights; one for each subset 
	'''
	assert issparse(C), "cover must be sparse matrix"
	n,J = C.shape
	assert len(W) == J, "Number of weights must match number of subsets"
	w = W.reshape((len(W), 1)) # ensure w is a column vector
	# linprog(w, A_ub)


# Greedy provides an H_k-approximation for the weight k set cover, where H_k = \sum\limits_{i=1}^k 1/i is the k-th harmonic number
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.1615&rep=rep1&type=pdf
def greedy_weighted_set_cover(n, S, W):
	''' 
	Computes a set of indices I \in [m] whose subsets S[I] = { S_1, S_2, ..., S_k }
	yield an approximation to the minimal weighted set cover, i.e.

		S* = argmin_{I \subseteq [m]} \sum_{i \in I} W[i]
				 such that S_1 \cup ... \cup S_k covers [n]
	
	Parameters: 
		n: int := The number of points the subsets must cover 
		S: (n x J) sparsematrix := A sparse matrix whose non-zero elements indicate the subsets (one subset per column-wise)
		W: ndarray(J,1) := weights for each subset 

	Returns: 
		C: ndarray(k,1) := set of indices of which subsets form a minimal cover 
	'''
	assert issparse(S)
	assert S.shape[0] == n and S.shape[1] == len(W)
	J = S.shape[1]

	def covered(I):
		membership = np.zeros(n, dtype=bool)
		for j in I: membership[np.flatnonzero(S[:,j].A)] = True
		return(membership)

	C = []
	membership = covered(C)
	while not(np.all(membership)):
		not_covered = np.flatnonzero(np.logical_not(membership))
		cost_effectiveness = []
		for j in range(J):
			S_j = np.flatnonzero(S[:,j].A)
			size_uncovered = len(np.intersect1d(S_j, not_covered))
			cost_effectiveness.append(size_uncovered/W[j])
		C.append(np.argmax(cost_effectiveness))
		# print(C[-1])
		membership = covered(C)
	
	## Return the greedy cover
	return(np.array(C))
