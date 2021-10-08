# Topological Assembly of Local Euclidean Models 

This repository implements TALLEM - a topologically inspired non-linear dimensionality reduction method.

Given some data set *X* and a map <img class='latex-inline math' style="background: white; vertical-align:-0.105206pt;" src="https://render.githubusercontent.com/render/math?math=\large f%20%3A%20X%20%5Cto%20B&mode=inline"> onto some topological space _B_ which captures the topology/nonlinearity of _X_, TALLEM constructs a map <img style="background: white; vertical-align:-0.105206pt" class='latex-inline math' src="https://render.githubusercontent.com/render/math?math=\large F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20&mode=inline"> mapping _X_ to a _D_-dimensional space. 

TODO: describe TALLEM more

## Dependencies 

`tallem` requires _Python >= 3.9.1_, along with the packages listed in [pyproject.toml](https://github.com/peekxc/tallem/blob/a1e7d2cd5d0dab5816ece658a3816dc0425f2391/pyproject.toml#L12). These are automatically downloaded and installed via `pip` using the installation procedure given below.

Externally, `tallem` uses [pybind11](https://github.com/pybind/pybind11/tree/stable) to interface with a variety of software libraries using [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), which themselves must be installed in order to run TALLEM. These include: 

* [Armadillo](http://arma.sourceforge.net/) >= 10.5.2
* [CARMA](https://github.com/RUrlus/carma) >= v0.5
* [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/) (for building the [extension modules](https://docs.python.org/3/glossary.html#term-extension-module))

Since prebuilt wheels are not yet provided, a [C++17 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support/17) may be needed to install these dependencies. 

## Installing

Currently, `tallem` must be built from source--wheels will be made available on PyPI or some other host in the future. 

Meson and Ninja are installeable with `pip`:

```bash
pip install meson ninja 
```

Armadillo [provides a variety of installation options](http://arma.sourceforge.net/download.html).

CARMA is a [header-only](https://en.wikipedia.org/wiki/Header-only), the source files only require the directory where the files are requires building from source using [CMAKE](https://cmake.org/runningcmake/). On UNIX-like terminals, this can be achieved via: 

```bash
git clone https://github.com/RUrlus/carma
cd carma | cmake . | make | sudo make install 
```

Ensure the path to CARMA in the `meson.build` script matches where it was installed (e.g. `/usr/local/carma/include`). 

`tallem` can be built using [`build`](https://pypa-build.readthedocs.io/en/stable/) package builder:

```bash
python -m mesonbuild.mesonmain build
meson install -C build
python -m build 
```

Assuming this succeeds, the [wheel](https://packaging.python.org/glossary/#term-Wheel) should be located in the `dist` folder, from which it can be installed the local [site-packages](https://docs.python.org/3/library/site.html#site.USER_SITE) via: 

```bash
pip install dist/tallem-*.whl
```

If you have an installation problems or questions, feel free to [make a new issue](https://github.com/peekxc/tallem/issues).

## Usage 

Below is some example code showcasing TALLEMs ability to handle topological obstructions to dimensionality reduction like non-orientability.  

```python
from tallem import TALLEM
from tallem.cover import IntervalCover
from tallem.datasets import mobius_band

## Get mobius band data + its parameter space
X, B = mobius_band(n_polar=26, n_wide=6, embed=3).values()
B_polar, B_radius = B[:,[1]], B[:,[0]]

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

**FAQ**



_The dependencies listed require Python 3.5+, but I'm using an older version of Python. Will`tallem` still run on my machine, and if not, how can I make `tallem` compatible?_

`tallem` requires Python version 3.5 or higher and will not run on older versions of Python. If your version of Python is older than this, consider installing `tallem` in a [virtual environment] that supports Python 3.5+. 
Alternatively, you're free to make the appropriate changes to `tallem` to make the library compatible with an older version yourself and then issue a PR. 

