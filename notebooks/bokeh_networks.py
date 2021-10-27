# %% Network
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, from_networkx
from bokeh.models import GraphRenderer, Ellipse, Range1d, Circle, ColumnDataSource, MultiLine, Label, LabelSet, Button
from bokeh.palettes import Spectral8, RdGy
from bokeh.models.graphs import StaticLayoutProvider
from bokeh.io import output_notebook, show, save
from bokeh.transform import linear_cmap
from bokeh.layouts import column

output_notebook()

# Compute hausdorff distance + cmds 
# from scipy.spatial.distance import directed_hausdorff
# from itertools import combinations
# index_set = list(top.cover.keys())
# d_h1 = np.array([directed_hausdorff(X[top.cover[i],:], X[top.cover[j],:])[0] for i,j in combinations(index_set, 2)])
# d_h2 = np.array([directed_hausdorff(X[top.cover[j],:], X[top.cover[i],:])[0] for i,j in combinations(index_set, 2)])
# d_H = np.maximum(d_h1, d_h2)
# layout = cmds(d_H**2)


G = nx.Graph()
G.add_nodes_from(range(len(top.cover)))
G.add_edges_from(top.alignments.keys())

# Spring layout
v_coords = np.array(list(nx.spring_layout(G).values()))
v_sizes = np.array([len(subset) for index, subset in top.cover.items()])

x_rng = np.array([np.min(v_coords[:,0]), np.max(v_coords[:,0])])*[0.90, 1.10]
y_rng = np.array([np.min(v_coords[:,1]), np.max(v_coords[:,1])])*[0.90, 1.10]

#Create a plot — set dimensions, toolbar, and title
p = figure(
	tools="pan,wheel_zoom,save,reset", 
	active_scroll='wheel_zoom',
	x_range=x_rng, 
	y_range=y_rng, 
	title="TALLEM Nerve complex"
)
edge_x = [v_coords[e,0] for e in G.edges]
edge_y = [list(v_coords[e,1]) for e in G.edges]

from tallem.color import bin_color, colors_to_hex, linear_gradient

align_edge_color = bin_color(alignment_error, linear_gradient(["gray", "red"], 100)['hex'])
frame_edge_color = bin_color(list(frame_error.values()), linear_gradient(["gray", "red"], 100)['hex'])

p.multi_line(edge_x, edge_y, color=frame_edge_color, alpha=0.80, line_width=4)

p.circle(v_coords[:,0], v_coords[:,1], size=v_sizes*0.15, color="navy", alpha=1.0)

p.toolbar.logo = None
p.toolbar_location = None
show(p)



# button = Button(label="Assemble", button_type="primary", width_policy="min")
# button.on_click(lambda event: print("Assembling frames"))

# gridplot([[s1, s2, s3]], toolbar_location=None)
## Get notion of how high the procrustes error scales too 
from scipy.linalg import orthogonal_procrustes
n = 1000
X = np.random.uniform(size=(n,2), low = 0.0, high = 10)
Y = np.random.uniform(size=(n,2), low = 0.0, high = 10)

# Translation
x_centroid, y_centroid = X.mean(0), Y.mean(0) # centroids
X, Y = X - x_centroid, Y - y_centroid        # center

# Scaling 
XS, YS = np.linalg.norm(X), np.linalg.norm(Y)
X /= XS
Y /= YS

R, sca = orthogonal_procrustes(X, Y)
print(np.linalg.norm(X @ R - Y))



## Alignment error
alignment_error = np.array([a['distance'] for a in top.alignments.values()])
# max here: 2


## Get error between Phiframes 
frame_error = {}
index_set = list(top.cover.keys())
for ((j,k), pa) in top.alignments.items():
	omega_jk = pa['rotation'].T
	X_jk = np.intersect1d(top.cover[index_set[j]], top.cover[index_set[k]])
	frame_error[(j,k)] = 0.0
	for x in X_jk:
		phi_j = top._stf.generate_frame(j, np.ravel(top.pou[x,:].A))
		phi_k = top._stf.generate_frame(k, np.ravel(top.pou[x,:].A))
		frame_error[(j,k)] += np.linalg.norm((phi_j @ omega_jk) - phi_k)
	frame_error[(j,k)] = frame_error[(j,k)]/len(X_jk)

# max here: 2*np.sqrt(2)

# Linked brushing in Bokeh is expressed by sharing data sources between glyph renderers.

from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from tallem.dimred import cmds

X = np.random.uniform(size=(15,3), low=0, high=10.0)
Y = X + np.random.uniform(size=(15,3))
x = cmds(X, 2)
y = cmds(Y, 2)

fig, ax = plt.subplots()
ax.scatter(*y.T)
ax.scatter(*x.T)

angle = 3*np.pi/4
R = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
z = ((R @ y.T) * 3.5).T

ax.scatter(*x.T)
ax.scatter(*y.T)

z = (R @ x.T).T

from scipy.linalg import orthogonal_procrustes
R_pro, d_pro = orthogonal_procrustes(x, z) 


procrustes(x, z)

# source = ColumnDataSource(data=
# 	dict(
# 		height=[66, 71, 72, 68, 58, 62],
# 		weight=[165, 189, 220, 141, 260, 174],
# 	  names=['Mark', 'Amir', 'Matt', 'Greg','Owen', 'Juan']
# 	))
# labels = LabelSet(x='weight', y='height', text='names', x_offset=5, y_offset=5, source=source, render_mode='canvas')
# p.add_layout(labels)
# show(p)

#Create a network graph object with spring layout
# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
# network_graph = from_networkx(G, nx.spring_layout, scale=10, center=(0, 0))

#Set node size and color
# network_graph.node_renderer.glyph = Circle(size=15, fill_color='skyblue', color=col)


# col = list(np.repeat("firebrick", len(top.cover)-1)).append('skyblue')

# #Set edge opacity and width
# # network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=5)
# #network_graph.edge_renderer.glyph.line_color = "firebrick"
# #e_map = linear_cmap(field_name='alignment', palette=RdGy ,low=0.0, high=15.0)

# np.max(G.edges)

# #Add network graph to the plot
# plot.renderers.append(network_graph)


theta = np.linspace(0, 2*np.pi, 100)
circle = np.hstack((np.cos(theta), np.sin(theta)))

from tallem import TALLEM
from tallem.dimred import *
from tallem.cover import *
from tallem.datasets import *

## Run TALLEM on interval cover using polar coordinate information
m_dist = lambda x,y: np.minimum(abs(x - y), (2*np.pi) - abs(x - y))
cover = IntervalCover(theta, n_sets = 15, overlap = 0.40, space = [0, 2*np.pi], metric = m_dist)
embedding = TALLEM(cover, local_map="pca2", n_components=3).fit_transform(X, theta)

## Rotate and view
angles = np.linspace(0, 360, num=12, endpoint=False)
scatter3D(embedding, angles=angles, figsize=(16, 8), layout=(2,6), c=polar_coordinate)




# %% Detecting the existence of multiple clusters 
import numpy as np
import matplotlib.pyplot as plt
from tallem.dimred import knn_graph

X0 = np.random.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2)*0.30, size=50)
X1 = np.random.multivariate_normal(mean=[3.0, 1.0], cov=np.eye(2)*0.025, size=30)
X2 = np.random.multivariate_normal(mean=[-1.0, 1.0], cov=np.eye(2)*0.025, size=30)

Z = np.vstack((X0, X1, X2))
plt.scatter(*Z.T)


Z = np.vstack((X0, X1))
G = knn_graph(Z, k = 15)

knn_dist = np.max(G.A, axis = 0)
knn_dens = (15/Z.shape[0])*(1/(knn_dist**Z.shape[1]))

from scipy.sparse.csgraph import dijkstra
max_ind = np.argmax(knn_dens)
d, p = dijkstra(G, directed=False, indices=max_ind, return_predecessors=True)

min_path_density = []
for v in range(len(p)):
	path = [v]
	while p[v] != -9999:
		path.append(p[v])
		v = p[v]
	min_path_density.append(np.min(knn_dens[path]))
min_path_density = np.array(min_path_density)

r_size = min_path_density/knn_dens


plt.scatter(*X0.T, c="red")
plt.scatter(*X1.T, c="blue")
plt.scatter(*Z[max_ind,:], c="green")

plt.hist(knn_dens[:50])
plt.hist(knn_dens[50:])

plt.hist(min_path_density[:50], color="yellow")
plt.hist(min_path_density[50:], color="yellow")

plt.hist(r_size[:50], color="red")
plt.hist(r_size[50:], color="red")

## Low average => probably just one cluster 
np.mean(1.0 - r_size[50:]) 


np.mean(1.0 - r_size[:50]) 

