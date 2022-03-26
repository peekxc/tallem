## TALLEM __init__.py file 
## Imports all the main utilities of TALLEM and provides a wrapper for the entire function

## Base external imports 
import re
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import Callable, Iterable, List, Set, Dict, Optional, Tuple, Any, Union, Sequence
from itertools import combinations
from scipy.sparse import issparse, csc_matrix

## Import the extension modules subpackage
import tallem.extensions

# TODO: define __all__
# __all__ = [
# 	'function_1',
# 	'function_2'
# ]

## Relative imports ( tallem-specific )
from .utility import find_where
from .sc import delta0D
from .distance import dist
from .cover import CoverLike, IntervalCover, partition_of_unity, validate_cover, is_uniform
from .alignment import opa, align_models, global_translations
from .samplers import uniform_sampler
from .dimred import *
from .stiefel import frame_reduction
from .assembly import assemble_frames, assembly_fast

class TALLEM():
	'''
	TALLEM: Topological Assembly of Locally Euclidean Models

	__init__(cover, local_map, D, pou) -> (TALLEM instance): 
		cover := CoverLike iterable of length J that covers some topological space B, where B is the image of some map f : X -> B
		local_map := a callable mapping (m x p) subsets of X to some (m x d) space, where d < p, which approximately preserves the metric on X. 
		D := target embedding dimension
		pou := partition of unity, either one of ['triangular', 'quadratic'], or an (n x J) ArrayLike object whose rows
					 yield weights indicating the strength of membership of that point with each set in the cover.

	fit(X, pou) -> (TALLEM instance): 
		X := an (n x p) numpy array representing *n* points in *p* space.
		B := an (n x q) numpy array representing the image of f : X -> B, where f is a map that cpatures the topology and non-linearity of X. 

	fit_transform(X, kwargs) -> ArrayLike:
		X := an (n x p) numpy array representing *n* points in *p* space.
		B := an (n x q) numpy array representing the image of f : X -> B
		kwargs := extra arguments passed to .fit(**kwargs)

	transform() -> ArrayLike (TODO)

	'''
	
	def __init__(self, cover: CoverLike, local_map: Union[str, Callable[npt.ArrayLike, npt.ArrayLike]], D: int = 2, pou: Union[str, csc_matrix] = "default"):
		# assert isinstance(cover, CoverLike), "cover argument must be an CoverLike."
		
		## Store callable as part of the cover's initialization
		self.D = D ## Target dimension of the embedding 

		## If string supplied to shortcut dim. reduction step, parse that, otherwise expect it to be a callable
		if isinstance(local_map, str):
			dr_methods = ["mmds", "cmds", "nmds", "pca", "iso", "lle", "le", "hlle"]
			method, d = re.findall(r'(\w+)(\d+)', local_map)[0]
			assert method in dr_methods, "Invalid method supplied"
			def dr_function(f, d): 
				return(lambda x: f(x, d))
			local_f = [mmds, cmds, nmds, pca, isomap][dr_methods.index(method)]
			self.local_map = dr_function(local_f, int(d))
		else:
			assert isinstance(local_map, Callable), "local model map must be a function."
			self.local_map = local_map
		
		## Assign cover as-is, validate on fit
		self.cover = cover 

		## Validate the partition of unity respects the closure condition relative to the cover 
		if isinstance(pou, csc_matrix): assert pou.shape[1] == len(self.cover), "Partition of unity must have one column per element of the cover"
		self.pou = "identity" if (isinstance(pou, str) and pou == "default") else pou

	def fit(self, X: ArrayLike, pou: Optional[Union[str, csc_matrix]] = None, **kwargs):
		if pou is None: pou = self.pou
		
		## Validate cover
		self.n = inverse_choose(len(X), 2) if is_pairwise_distances(X) else X.shape[0]
		assert validate_cover(self.n, self.cover), "Invalid cover -- does not fully cover X."
		if not(is_uniform(self.cover)):
			import warnings
			warnings.warn('Cover subsets appear to be highly non-uniform or sparse -- embedding may be low quality.', RuntimeWarning)

		## Map the local euclidean models (in parallel)
		self.models = fit_local_models(self.local_map, X, self.cover)

		## Construct a partition of unity
		self.pou = pou if issparse(pou) else partition_of_unity(cover = self.cover, mollifier = pou)
		assert issparse(self.pou), "partition of unity must be a sparse matrix"

		## Align the local reference frames using Procrustes
		self.alignments = align_models(self.cover, self.models, **kwargs)
		
		## Get global translation vectors using cocyle condition
		self.translations = global_translations(self.cover, self.alignments)

		## Solve the Stiefel manifold optimization for the projection matrix 
		self.A0, self.A, self._stf = frame_reduction(self.alignments, self.pou, self.D, **kwargs)

		## Assemble the frames!
		## See: https://github.com/rasbt/python-machine-learning-book/blob/master/faq/underscore-convention.md
		self.embedding_ = assembly_fast(self._stf, self.A, self.cover, self.pou, self.models, self.translations)
		
		## Save useful information
		self.d, self.D = self._stf.d, self.A0.shape[1]
		return(self)


	## TODO: somehow extend everything to support arbitrary inputs for out-of-sample extension
	## Idea: Iterables.... instead of covers... or as optional arguments on top of covers...
	## Cover should probably support operator(..., index) as a set_contains 
	# def transform(self, B: ArrayLike, pou: Optional[csc_matrix] = None, D_frame: Optional[npt.ArrayLike] = None) -> ArrayLike:
	# 	## Map new data points in 'B' based on the existing cover and local map
	# 	extend_models = fit_local_models(self.local_map, B, self.cover)
	# 	extend_pou = pou if issparse(pou) else partition_of_unity(cover = self.cover, mollifier = pou, points...)
	# 	assert issparse(pou), "partition of unity must be a sparse matrix"
	# 	extend_alignments = align_models(self.cover, self.models, points...)
	# 	extend_translations = global_translations(self.cover, self.alignments, points...)
	# 	self.assemble(extend_pou, self.A)

	## Here, B would have to match the elements the cover was constructed on
	def fit_transform(self, X: npt.ArrayLike, **kwargs) -> npt.ArrayLike:		
		self.fit(X, self.pou, **kwargs)
		return(self.embedding_)

	def __repr__(self) -> str:
		return("TALLEM instance")

	def assemble(self, pou: Optional[csc_matrix] = None, D_frame: Optional[npt.ArrayLike] = None) -> npt.ArrayLike:
		''' 
		Assembles the local models in a lower-dimensional (D) space using 'D_frame' 

		Parameters: 
			pou := partition of unity to use with the assembly. Defaults to the precomputed one. See details. 
			D_frame := frame to use to project down to D dimensions

		As before, the partition of unity must respect the closure property with respect to the cover. 
		'''
		if D_frame is None: D_frame = self.A
		if pou is None: pou = self.pou
		return(assembly_fast(self._stf, D_frame, self.cover, pou, self.models, self.translations, False))

	def assemble_high(self, pou: Optional[csc_matrix] = None) -> npt.ArrayLike:
		''' 
		Assembles the local models into a high-dimensional (dJ) space. 
		This step is usually done per-point prior to projecting down to a lower-dimensional space. 
		It can be useful to have the high-dimensional coordinates in certain debugging situations (e.g. 
		assessing the quality of the (dJ -> d) projection).

		Parameters: 
			pou := partition of unity to use with the assembly. Defaults to the precomputed one. See details. 

		As before, the partition of unity must respect the closure property with respect to the cover. 
		'''
		if pou is None: pou = self.pou
		return(assembly_fast(self._stf, self.A, self.cover, pou, self.models, self.translations, True))

	def nerve_graph(self):
		ask_package_install("networkx")
		import networkx as nx
		G = nx.Graph()
		G.add_nodes_from(range(len(self.cover)))
		G.add_edges_from(self.alignments.keys())
		return(G)

	def build_nerve(self, build_local_models: bool = False):
		if build_local_models:
			self.models = fit_local_models(self.local_map, X, self.cover)
		assert self.models is not None and len(self.models) > 0
		self.alignments = align_models(self.cover, self.models)
		self.translations = global_translations(self.cover, self.alignments)

	def plot_nerve(self, 
		X: Optional[ArrayLike] = None, 
		layout=["hausdorff", "spring"], edge_color=["alignment", "frame"], 
		vertex_scale: float = 15.0, edge_scale: float = 8.0,
		toolbar = True, 
		notebook=True,
		**kwargs
	):
		from bokeh.plotting import figure, show, from_networkx
		from bokeh.models import GraphRenderer, Ellipse, Range1d, Circle, ColumnDataSource, MultiLine, Label, LabelSet, Button
		from bokeh.palettes import Spectral8, RdGy
		from bokeh.models.graphs import StaticLayoutProvider
		from bokeh.io import output_notebook, show, save
		from bokeh.transform import linear_cmap
		from bokeh.layouts import column
		if (notebook): output_notebook(verbose=False, hide_banner=True)
		G = self.nerve_graph()
		ec = np.ones((len(G.edges),), dtype=float)
		if (isinstance(edge_color, Iterable) and edge_color == ["alignment", "frame"]) or (isinstance(edge_color, str) and edge_color == "alignment"):
			## Alignment error (maximum == 2, as || A @ R - B ||_F <= |A|+|B|)
			ec = np.array([a['distance'] for a in self.alignments.values()])
			ec = ec / 2
		elif isinstance(edge_color, str) and edge_color == "frame":
			## Get error between Phiframes 
			frame_error = {}
			index_set = list(self.cover.keys())
			for ((j,k), pa) in self.alignments.items():
				omega_jk = pa['rotation'].T
				X_jk = np.intersect1d(self.cover[index_set[j]], self.cover[index_set[k]])
				frame_error[(j,k)] = 0.0
				for x in X_jk:
					phi_j = self._stf.generate_frame(j, np.ravel(self.pou[x,:].A))
					phi_k = self._stf.generate_frame(k, np.ravel(self.pou[x,:].A))
					frame_error[(j,k)] += np.linalg.norm((phi_j @ omega_jk) - phi_k)
				frame_error[(j,k)] = frame_error[(j,k)]/len(X_jk)
				ec = np.array(list(frame_error.values()))
				ec = ec / 2*np.sqrt(2)
		elif isinstance(edge_color, str) and edge_color == "translation":
			taus = self.translations
			d = len(taus[0])
			translation_error = {}
			for (j,k) in self.alignments.keys():
				omega_jk = self.alignments[(j,k)]['rotation']
				tau_j, tau_k = taus[j].reshape((d, 1)), taus[k].reshape((d, 1))
				tau_jk = self.alignments[(j,k)]['translation']
				tau_jk_approx = (omega_jk @ tau_j) - tau_k
				tau_error = np.linalg.norm(tau_jk_approx - tau_jk)
				tau_size_ratio = np.max([np.linalg.norm(tau_jk_approx)/np.linalg.norm(tau_jk), np.linalg.norm(tau_jk_approx)/np.linalg.norm(tau_jk)])
				translation_error[(j,k)] = np.min([tau_error*tau_size_ratio, 5.0]) # 5x ratio should be quite bad
			ec = np.array(list(translation_error.values()))/5.0
		else: 
			raise ValueError("Invalid edge metric")

		from scipy.spatial.distance import directed_hausdorff
		from itertools import combinations
		from tallem.dimred import cmds
		use_grid_lines = False
		if (X is None) or isinstance(layout, str) and layout == "spring":
			import networkx as nx
			layout = np.array(list(nx.spring_layout(G).values()))
		elif (isinstance(layout, str) and layout == "hausdorff"):
			assert isinstance(X, np.ndarray)
			index_set = list(self.cover.keys())
			d_h1 = np.array([directed_hausdorff(X[self.cover[i],:], X[self.cover[j],:])[0] for i,j in combinations(index_set, 2)])
			d_h2 = np.array([directed_hausdorff(X[self.cover[j],:], X[self.cover[i],:])[0] for i,j in combinations(index_set, 2)])
			d_H = np.maximum(d_h1, d_h2)
			layout = cmds(d_H**2)
			use_grid_lines = True
		else:
			raise ValueError("Unimplemented layout")
			
		## Vertex sizes == size of each preimage
		v_sizes = np.array([len(subset) for index, subset in self.cover.items()])
		v_widths = (v_sizes / np.max(v_sizes))*vertex_scale

		# edge_sizes = {}
		# for i,j in top.alignments.keys():
		# 	ij_ind = np.intersect1d(cover[i], cover[j], return_indices=False)
		# 	edge_sizes[(i,j)] = len(ij_ind)

		TOOLTIPS = [
				("index", "$index"),
				("size", "@n"),
		]



		#Create a plot — set dimensions, toolbar, and title
		x_rng = np.array([np.min(layout[:,0]), np.max(layout[:,0])])*[0.90, 1.10]
		y_rng = np.array([np.min(layout[:,1]), np.max(layout[:,1])])*[0.90, 1.10]
		p = figure(
			tools="pan,wheel_zoom,lasso_select,reset", 
			active_scroll=None,
			active_drag="auto",
			x_range=x_rng, 
			y_range=y_rng, 
			title="Nerve complex of the Cover", 
			tooltips=TOOLTIPS,
			plot_width=300, 
			plot_height=300,
			**kwargs
		)
		p.axis.visible = False
		p.xgrid.visible = use_grid_lines
		p.ygrid.visible = use_grid_lines
		edge_x = [layout[e,0] for e in G.edges]
		edge_y = [list(layout[e,1]) for e in G.edges]

		## Edge widths
		index_set = list(self.cover.keys())
		e_sizes = []
		for ((j,k), pa) in self.alignments.items():
			X_jk = np.intersect1d(self.cover[index_set[j]], self.cover[index_set[k]])
			e_sizes.append(len(X_jk))
		e_sizes = np.array(e_sizes)
		e_widths = (e_sizes / np.max(e_sizes))*edge_scale

		from tallem.color import bin_color, colors_to_hex, linear_gradient
		ec = bin_color(ec, linear_gradient(["gray", "red"], 100)['hex'], min_x = 0.0, max_x=1.0)

		edge_data = {
			'xs' : edge_x,
			'ys' : edge_y,
			'color' : ec,
			'line_width': e_widths,
			'n' : e_sizes
		}
		edge_source = ColumnDataSource(data=edge_data)

		p.multi_line('xs', 'ys', color='color', line_width='line_width', alpha=0.80, source=edge_source)
		
		node_data = {
			'x' : layout[:,0],
			'y' : layout[:,1],
			'size' : v_widths,
			'n' : v_sizes
		}
		node_source = ColumnDataSource(data=node_data)

		p.circle('x', 'y', size='size', color="navy", alpha=1.0, source=node_source)

		p.toolbar.logo = None
		if toolbar == False: p.toolbar_location = None
		show(p)


	def _profile(self, **kwargs):
		ask_package_install("line_profiler")
		import line_profiler
		profile = line_profiler.LineProfiler()
		profile.add_function(self.fit)
		profile.add_function(partition_of_unity)
		profile.add_function(frame_reduction)
		profile.add_function(assembly_fast)
		profile.add_function(align_models)
		profile.enable_by_count()
		self.fit(**kwargs)
		profile.print_stats(output_unit=1e-3)
	

# %% AGA-tallem function
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def gamma_tau(top, node_path, verbose=0):
	Gamma = np.eye(top.d)
	for cc in range(len(node_path)-1):
		k,l = node_path[cc], node_path[cc+1]
		omega_k = top.alignments[(k, l)]['rotation']
		Gamma = Gamma @ omega_k
		if verbose > 0: print(f"({k}, {l})", end='')
	if verbose > 0: print("")
	return(Gamma)

def aga_tallem(top, X):

	## Start with MST 
	import networkx as nx
	G = nx.Graph()
	G.add_nodes_from(range(len(top.cover)))
	for (j,k) in top.alignments.keys():
		d_jk = directed_hausdorff(X[top.cover[j],:], X[top.cover[k],:])[0]
		d_kj = directed_hausdorff(X[top.cover[k],:], X[top.cover[j],:])[0]
		G.add_edge(j, k, weight=np.maximum(d_jk, d_kj))
	T = nx.minimum_spanning_tree(G) 

	## Compute gammas + taus
	Gammas, Taus = [], np.zeros(shape=(len(top.cover), top.d))
	root_node = 0
	for j in range(len(top.cover)):
		node_path = nx.dijkstra_path(T, root_node, j)
		omega_path = []
		for cc in range(len(node_path)-1):
			k, l = node_path[cc], node_path[cc+1]
			omega_lk = top.alignments[(l,k)]['rotation']
			v_lk = top.alignments[(l,k)]['translation']
			omega_path.append(omega_lk)
			Taus[l,:] = omega_lk @ Taus[k,:] - v_lk
		Gammas.append(gamma_tau(top, node_path)) 

	## Restrict data set to G / T 
	T_edges = list(T.edges())
	Y_ind = np.array(list(range(X.shape[0])))
	for (j,k) in G.edges:
		if not((j,k) in T_edges):
			jk_ind = np.intersect1d(top.cover[j], top.cover[k])
			Y_ind = np.setdiff1d(Y_ind, jk_ind)

	## Compute the assembly 
	Z = np.zeros((len(Y_ind), top.d))
	for i, idx in enumerate(Y_ind):
		vp = top.pou[idx,:].A.flatten()
		pou_ind = np.flatnonzero(vp)
		for j in pou_ind: 
			lm_point_ind = np.searchsorted(top.cover[j], idx)
			f_xi = top.models[j][lm_point_ind,:]
			Z[i,:] += vp[j]*Gammas[j] @ (f_xi+Taus[j,:])

	return(Z, T, Y_ind)