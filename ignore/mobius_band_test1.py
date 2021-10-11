# %% 
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

# %% Mobius band example
import numpy as np
from src.tallem import TALLEM
from src.tallem.cover import IntervalCover
from src.tallem.datasets import mobius_band





#%% 
## Generate mobius band + polar coordinate 
X, B = mobius_band(n_polar=120, n_wide=15, scale_band = 0.25)
polar_coordinate = B[:,[1]]

#%% 
## Assemble the embedding with TALLEM
m_dist = lambda x,y: np.sum(np.minimum(abs(x - y), (2*np.pi) - abs(x - y)))
cover = IntervalCover(polar_coordinate, n_sets = 20, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)

# from src.tallem.dimred import isomap
# local_map = lambda x: isomap(x, d=3, k=15)

top = TALLEM(cover, local_map="pca2", n_components=3)
emb = top.fit_transform(X=X, B=B)

# %% 
top._profile(X=X, B=B)

#%% 
%%time
from src.tallem.dimred import fit_local_models
M = fit_local_models(top.local_map, X, top.cover, 8)

#%% 

ind = list(top.cover.values())
with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
	future = executor.map(np.sum, ind)
	for ind_sum in future:
		print(ind_sum)

# models = {}
# do_euclidean_model = lambda ce: (ce[0], top.local_map(X[np.array(ce[1]),:])) 
# with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
# 	future = executor.map(do_euclidean_model, top.cover.items())
# 	for index, model in future:
# 		models[index] = model
from multiprocessing import Pool, cpu_count
def main():
	hard_work = lambda x: np.sqrt(x) 
	try:
			workers = cpu_count()
	except NotImplementedError:
			workers = 1
	pool = Pool(processes=workers)
	result = pool.map(hard_work, range(100, 1000000))

#%%
## Visualize the resulting embedding
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(*emb.T, c = B)


# %% Debugging 
top = TALLEM(cover, local_map="pca2", n_components=3)
# top.fit(X, B, pou="triangular")
top._profile(X=X, B=B)

# %% 
import line_profiler
profile = line_profiler.LineProfiler()
def do_func():
	c = 0
	for i in range(1000): c += i
	#if not('profile' in vars()) and not('profile' in globals()):
	for i in range(1000): c += np.sqrt(np.cos(i*5))
	return(c)
profile.add_function(do_func)
profile.enable_by_count()
do_func()
profile.print_stats(output_unit=1e-3)


# %% 
iota = np.array(top.pou.argmax(axis=1)).flatten()
pou_t = top.pou.transpose().tocsc()

# %% Sparse population of frames
%%time
top._stf.populate_frames(iota, pou_t, True) 
# Wall time: 32.7 s

# %% Dense population of frames
%%time
top._stf.populate_frames(iota, pou_t, False) 
# Wall time: 95.5 ms

# %% Sparse population of frames (2nd attempt)
iota = np.array(top.pou.argmax(axis=1)).flatten()
pou_t = top.pou.transpose().tocsc()

#%% Dense population
%%time
iota = np.array(top.pou.argmax(axis=1)).flatten()
top._stf.populate_frames(iota, pou_t, False)

#%% 
%%time
top._stf.populate_frames_sparse(pou_t)

#%%
%%time
ew, ev = top._stf.initial_guess(top._stf.D, True)

#%% 
Fb = top._stf.all_frames() ## Note these are already weighted w/ the sqrt(varphi)'s!

#%% 
%%time 
Eval, Evec = np.linalg.eigh(Fb @ Fb.T)
A0 = Evec[:,np.argsort(-Eval)[:D]]

#%% 
# top._stf.populate_frames_sparse(iota, pou_t)
# top._stf.all_frames()
# top._stf.all_frames_sparse()
# 
# np.max(x.flatten())

from scipy.sparse import csc_matrix
i,j,x = top._stf.all_frames_sparse()
frames = csc_matrix((x.flatten(), i.flatten(), j.flatten()), shape=(top._stf.d*len(top.cover), top._stf.d*top._stf.n))
frames_dense = top._stf.all_frames()

frames[:,0:5].A
frames_dense[:,0:5]

top.alignments

pou_t[:,0].A

top._stf.get_rotation(0, 1, len(top.cover))

np.sum(frames.A != frames_dense)

## even non-zero entry locations wrong
(frames.A != 0.0) == (frames_dense != 0.0)

fs = frames.A
[np.any(fs[:,j] != frames_dense[:,j]) for j in range(fs.shape[1])]

frames_dense[:,0]

# %% Solving for A0
%%time
Fb = top._stf.all_frames() ## Note these are already weighted w/ the sqrt(varphi)'s!
Eval, Evec = np.linalg.eigh(Fb @ Fb.T)
A0 = Evec[:,np.argsort(-Eval)[:3]]
# Wall time: 23.7 ms

# %% 
## Generate mobius band + polar coordinate 
M = mobius_band(n_polar=250, n_wide=15, scale_band = 0.25, plot=False, embed=6)
X, B = M['points'], M['parameters'][:,[1]]

## Assemble the embedding with TALLEM
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(B, n_sets = 50, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)

top = TALLEM(cover, local_map="pca2", n_components=3)
top._profile(X=X, B=B)


# %% 
%%time 
E = assembly_fast2(top._stf, top.A, top.cover, top.pou.transpose().tocsc(), top.models, top.translations)

# np.all(E == top.assemble().T)

# %% 
%%time
top.assemble()

## True!
np.all(top.fit_transform(X, B).T == E)

# %%
%%time 
n = X.shape[0]
for i in range(n): top._stf.populate_frame(i, np.sqrt(np.ravel(top.pou[i,:].todense())), False)
# f1 = top._stf.all_frames()
# %%
%%time 
iota = np.array(top.pou.argmax(axis=1)).flatten()
pou_t = top.pou.transpose().tocsc()
top._stf.populate_frames(iota, pou_t, False) # populate all the iota-mapped frames in vectorized fashion
# f2 = top._stf.all_frames()
