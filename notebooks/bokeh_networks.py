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
p.multi_line(edge_x, edge_y, color="firebrick", alpha=0.20, line_width=4)

p.circle(v_coords[:,0], v_coords[:,1], size=v_sizes*0.15, color="navy", alpha=0.5)

p.toolbar.logo = None
p.toolbar_location = None

button = Button(label="Assemble", button_type="primary", width_policy="min")

button.on_click(lambda event: print("Assembling frames"))

show(column(button, p))

# gridplot([[s1, s2, s3]], toolbar_location=None)

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
ax.scatter(*z.T)

from scipy.linalg import orthogonal_procrustes
orthogonal_procrustes(x, z)

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



## 
# mapper = linear_cmap(field_name='alignment', palette=RdGy ,low=min(y) ,high=max(y))

