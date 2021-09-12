import numpy as np
import numpy.typing as npt
from typing import Iterable, Dict
from scipy.sparse import csc_matrix
from .alignment import global_translations
from .utility import find_where
from .cover import CoverLike

def assemble_frames(stf, A: npt.ArrayLike, cover: CoverLike, pou: csc_matrix, local_models: Dict, translations: Dict) -> npt.ArrayLike:
	''' Performs the final assembly of all the frames. '''
	if len(translations) != len(cover): raise ValueError("There should be a translation vector for each subset of the cover.")
	assembly = np.zeros((stf.n, stf.D), dtype=np.float64)
	coords = np.zeros((1,stf.D), dtype=np.float64)
	index_set = list(local_models.keys())
	for i in range(stf.n):
		w_i = np.ravel(pou[i,:].todense())
		nz_ind = np.where(w_i > 0)[0]
		coords.fill(0)
		## Construct assembly functions F_j(x) for x_i
		for j in nz_ind: 
			subset_j = cover[index_set[j]]
			relative_index = find_where(i, subset_j, True) ## This should always be true!
			u, s, vt = np.linalg.svd((A @ (A.T @ stf.generate_frame(j, np.sqrt(w_i)))), full_matrices=False, compute_uv=True) 
			d_coords = local_models[index_set[j]][relative_index,:]
			coords += (w_i[j]*A.T @ (u @ vt) @ (d_coords + translations[j]).T).T
		assembly[i,:] = coords
	return(assembly)

def assembly_fast(stf, A: npt.ArrayLike, cover: CoverLike, pou: csc_matrix, local_models: Dict, translations: Dict) -> npt.ArrayLike:
	''' Performs the final assembly of all the frames. '''
	offsets = np.vstack([ offset for index,offset in translations.items() ])
	cover_subsets = [subset for index, subset in cover.items()]
	local_models = [coords for index, coords in local_models.items()]
	return(stf.assemble_frames(A, pou.tocsc(), cover_subsets, local_models, offsets))

