# %% Sampler imports
import numpy as np
import numpy.typing as npt
from tallem.landmark import landmarks

# %% Various samplers of the range [0,n)
def landmark_sampler(a: npt.ArrayLike, m, k, method: str = "precomputed"):
	''' 
	Sampler which uniformly samples from k precomputed landmark permutations 
	each of size k. 
	'''
	seed_idx = np.random.randint(0, a.shape[0], size=m)
	L = [landmarks(X, k, seed = si)['indices'] for si in seed_idx]
	g = np.random.default_rng()
	def sampler(n: int):
		nonlocal k, g, m, L
		num_rotations = int(n/k)
		num_extra = n % k 
		if num_rotations == 0:
			lm_idx = g.integers(0, m, size=1)[0] 
			for idx in L[lm_idx][:num_extra]:
				yield idx
		else: 
			full_lm_idx = g.integers(0, m, size=num_rotations) 
			extra_lm_idx = g.integers(0, m, size=1)[0] 
			for lm_idx in full_lm_idx:
				for idx in L[lm_idx]:
					yield idx
			for idx in L[extra_lm_idx][:num_extra]:
				yield idx
	return(sampler)

def uniform_sampler(n: int):
	g = np.random.default_rng()
	def sampler(k: int):
		nonlocal n, g
		for s in g.integers(0, n, size=k):
			yield s
	return(sampler)

def cyclic_sampler(n: int):
	current_idx = 0
	def sampler(num_samples: int):
		nonlocal current_idx
		for _ in range(num_samples):
			yield current_idx
			current_idx = current_idx+1 if (current_idx+1) < n else 0
	return(sampler)