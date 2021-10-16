# %% 
import sys
import os
PACKAGE_PARENT = '..'
sys.path.append(os.path.normpath(os.path.expanduser("~/tallem")))

# %% 
import numpy as np
from src.tallem import TALLEM
from src.tallem.cover import *
from src.tallem.datasets import mobius_band

#%% 
## Generate mobius band + polar coordinate 
X, B = mobius_band(n_polar=120, n_wide=15, scale_band = 0.25)
polar_coordinate = B[:,[1]]

#%% 
## Assemble the embedding with TALLEM
m_dist = lambda x,y: np.sum(np.minimum(abs(x - y), (2*np.pi) - abs(x - y)))
cover = IntervalCover(polar_coordinate, n_sets = 20, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)
assert validate_cover(X.shape[0], cover)

# %% 
top = TALLEM(cover, local_map="pca2", n_components=3)
assert isinstance(top, TALLEM)

#%% 
from src.tallem.dimred import fit_local_models
top.models = fit_local_models(top.local_map, X, top.cover)
assert isinstance(top.models, dict)
for model in top.models.values():
	assert isinstance(model, np.ndarray)

#%% Test partition of unity
from scipy.sparse import issparse
top.pou = partition_of_unity(polar_coordinate, cover = top.cover, similarity = top.pou)
assert issparse(top.pou)

# %% Align the local reference frames using Procrustes
from src.tallem.alignment import align_models
top.alignments = align_models(top.cover, top.models)
assert isinstance(top.alignments, dict)

# %% Get global translation vectors using cocyle condition
from src.tallem.alignment import  global_translations
top.translations = global_translations(top.cover, top.alignments)

# %% Solve the Stiefel manifold optimization for the projection matrix 
from src.tallem.stiefel import frame_reduction
from tallem.pbm import fast_svd
top.A0, top.A, top._stf = frame_reduction(top.alignments, top.pou, top.D)

# n, J, d = top.pou.shape[0], top.pou.shape[1], next(iter(top.models.values())).shape[1]
# stf = fast_svd.StiefelLoss(n, d, top.D)
# I1 = [index[0] for index in top.alignments.keys()]
# I2 = [index[1] for index in top.alignments.keys()]
# R = np.vstack([pa['rotation'] for index, pa in top.alignments.items()]) 
# stf.init_rotations(I1, I2, R, J)
# stf.setup_pou(top.pou.transpose().tocsc())
# iota = np.ravel(stf.extract_iota())
# stf.populate_frames_sparse(iota)
# ew, A0 = stf.initial_guess(top.D, True)

# %% Assembly 
from tallem.assembly import assembly_fast
top.embedding_ = assembly_fast(top._stf, top.A, top.cover, top.pou, top.models, top.translations)
# offsets = np.vstack([ offset for index,offset in translations.items()]).T
# cover_subsets = [np.sort(subset) for index, subset in cover.items()]
# local_models = [coords.T for index, coords in local_models.items()]
# return(stf.assemble_frames2(A, pou.transpose().tocsc(), cover_subsets, local_models, offsets).T)
assert isinstance(top.embedding_, np.ndarray)

# %% Run all 
embedding = top.fit_transform(X=X, B=polar_coordinate)

