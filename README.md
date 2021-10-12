# Topological Assembly of Local Euclidean Models 

This repository implements TALLEM - a topologically inspired non-linear dimensionality reduction method.

Given some data set *X* and a map <img class='latex-inline math' style="background: white; vertical-align:-0.105206pt;" src="https://render.githubusercontent.com/render/math?math=\large f%20%3A%20X%20%5Cto%20B&mode=inline"> onto some topological space _B_ which captures the topology/nonlinearity of _X_, TALLEM constructs a map <img style="background: white; vertical-align:-0.105206pt" class='latex-inline math' src="https://render.githubusercontent.com/render/math?math=\large F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20&mode=inline"> mapping _X_ to a _D_-dimensional space. 

TODO: describe TALLEM more

## Installing + Dependencies 

`tallem` requires _Python >= 3.8.0_. As a a [PEP 517](https://www.python.org/dev/peps/pep-0517/)-compliant package, the rest of the build requirements are listed in [pyproject.toml](https://github.com/peekxc/tallem/blob/main/pyproject.toml). These are automatically downloaded and installed via `pip` using the installation procedure given below.

###Installing from cibuildwheels

TODO

### Installing from source

To install `tallem` from source, clone the repository and install the package via: 

```bash
python -m pip install .
```

`tallem` relies on a few package dependencies in order to compile correctly when building from source. These libraries include: 

* [Armadillo](http://arma.sourceforge.net/) >= 10.5.2 ([see here for installation options](http://arma.sourceforge.net/download.html))
* [Poetry](https://python-poetry.org/) (for building the [source](https://packaging.python.org/glossary/#term-Source-Distribution-or-sdist) and [binary](https://packaging.python.org/glossary/#term-Wheel) distributions)
* [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/) (for building the [extension modules](https://docs.python.org/3/glossary.html#term-extension-module))

An install attempt of these external dependencies is made if they are not available prior to call to `pip`, however these may require manual installation. Additionally, the current source files are written in [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), so a [C++17 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support/17) will be needed. If you have an installation problems or questions, feel free to [make a new issue](https://github.com/peekxc/tallem/issues).

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

