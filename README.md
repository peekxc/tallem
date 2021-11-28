# Topological Assembly of Local Euclidean Models 

This repository implements TALLEM - a topologically inspired non-linear dimensionality reduction method.

Given some data set *X* and a map <img class='latex-inline math' style="background: white; vertical-align:-0.105206pt;" src="https://render.githubusercontent.com/render/math?math=\large f%20%3A%20X%20%5Cto%20B&mode=inline"> onto some topological space _B_ which captures the topology/nonlinearity of _X_, TALLEM constructs a map <img style="background: white; vertical-align:-0.105206pt" class='latex-inline math' src="https://render.githubusercontent.com/render/math?math=\large F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20&mode=inline"> mapping _X_ to a _D_-dimensional space. 

TODO: describe TALLEM more

## Dependencies 

`tallem`'s run-time dependencies are fairly minimal. They include:  

1. _Python >= 3.8.0_ 
2. *NumPy (>= 1.20)* and *SciPy* *(>=1.5)*

Package requirement details are listed in the [pyproject.toml](https://github.com/peekxc/tallem/blob/main/pyproject.toml). 
These are automatically downloaded using either of the installation methods described below.

Some functions which extend TALLEM's core functionality require additional dependencies to be called---they include *autograd*, *pymanopt*, *scikit-learn*, and *bokeh*. These packages are optional--they are not needed to get the embedding.

### Installing from pre-built wheels 

TODO: Make `tallem` pip-installeable by finishing wheel builds w/ cibuildwheels 

### Installing from source

The recommended way to build `tallem` distributions (source or built) is with [Poetry](https://python-poetry.org/). Once installed, navigate to `tallem`'s directory and use: 

```bash
poetry install -vvv
```

The default build script attempts to resolve all dependencies needed by the package at build-time. This includes possible source-installs of prerequisite 
C++ libraries and their associated build tools; `tallem` requires [Armadillo](http://arma.sourceforge.net/) (>= 10.5.2) for compilation of its [extension modules](https://docs.python.org/3/glossary.html#term-extension-module), whose builds are managed with [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/). Since these source files are written in [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), so a [C++17 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support/17) will be needed. 

If you plan on changing the code in any way, see the [developer note](#developer-note) about editeable installs. If you have an installation problems or questions, feel free to [make a new issue](https://github.com/peekxc/tallem/issues).

## Usage 

Below is some example code showcasing TALLEMs ability to handle topological obstructions to dimensionality reduction like non-orientability.  

```python
from tallem import TALLEM
from tallem.cover import IntervalCover
from tallem.datasets import mobius_band

## Get mobius band data + its parameter space
X, B = mobius_band()
B_polar = B[:,[1]]

## Construct a cover over the polar coordinate
m_dist = lambda x,y: np.sum(np.minimum(abs(x - y), (2*np.pi) - abs(x - y)))
cover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, metric = m_dist)

## Parameterize TALLEM + transform the data to the obtain the coordinization
emb = TALLEM(cover=cover, local_map="cmds2", n_components=3).fit_transform(X, B_polar)

## Draw the coordinates via 3D projection, colored by the polar coordinate
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*emb.T, marker='o', c=B_polar)
```

![mobius band](https://github.com/peekxc/tallem/blob/main/resources/tallem_polar.png?raw=true)



### Developer note

Installs from source using Poetry are done in [editeable mode](https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install) by default: thus one should be able to freely manipulate the source code once `tallem` is installed 
and see the changes immediately without restarting the session. If you're developing with Jupyter, be sure to add autoreload magics to the document: 

```python
%reload_ext autoreload
%autoreload 2 
# ... 
```

If you're developing with Microsoft's VSCode, you may want automating this via added the following to the settings.json file: 
```json
"jupyter.runStartupCommands": [
  "%load_ext autoreload", "%autoreload 2"
]
```
See [here](https://stackoverflow.com/questions/56059651/how-to-make-vscode-auto-reload-external-py-modules) for more details on this.

