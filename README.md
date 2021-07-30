# Topological Assembly of Local Euclidean Models 

This repository hosts `tallem`, a [Python project](https://packaging.python.org/glossary/#term-Project) which implements TALLEM--a topologically inspired non-linear dimensionality reduction method. Currently, `tallem` must be built from source, i.e. no wheels are available on PyPI yet. 

Given some data set *X* and a map <img style="background: white;" src="https://render.githubusercontent.com/render/math?math=f%20%3A%20X%20%5Cto%20B"> onto some topological space _B_ which captures the topology/nonlinearity of _X_, TALLEM constructs a map <img style="background: white;" src="https://render.githubusercontent.com/render/math?math=F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20"> mapping _X_ to a _D_-dimensional space. 

TODO: describe TALLEM more

## Dependencies 

`tallem` requires _Python >= 3.9.1_, along with the packages listed in [pyproject.toml](https://github.com/peekxc/tallem/blob/a1e7d2cd5d0dab5816ece658a3816dc0425f2391/pyproject.toml#L12). These are automatically downloaded and installed via `pip` using the installation procedure given below.

Outside of Python, `tallem` uses [pybind11](https://github.com/pybind/pybind11/tree/stable) to interface with a variety of software libraries using [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), 
which themselves must be installed in order to run TALLEM. These include: 

* [Armadillo](http://arma.sourceforge.net/) >= 10.5.2
* [CARMA](https://github.com/RUrlus/carma) >= v0.5
* [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/) (for building the [extension modules](https://docs.python.org/3/glossary.html#term-extension-module))

Since prebuilt wheels are not yet provided, a [C++17 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support/17) may be needed to install these dependencies. 

## Installing

Meson and Ninja are installeable with `pip`:

```bash
pip install meson ninja 
```

Armadillo [provides a variety of installation options](http://arma.sourceforge.net/download.html).

Though header-only, CARMA requires building from source using [CMAKE](https://cmake.org/runningcmake/). On UNIX-like systems, this can be achieved via: 

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

## The polar coordinate captures how the frames of reference rotate about B 
B_polar = B[:,1].reshape((B.shape[0], 1))

## The parameter space is discretized with a cover 
cover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, gluing=[1])

## Local euclidean models are specified with a function
f = lambda x: classical_MDS(dist(x, as_matrix=True), k = 2)

## Parameterize TALLEM + transform the data to the obtain the coordinization
embedding = TALLEM(cover=cover, local_map=f, n_components=3)
X_transformed = embedding.fit_transform(X, B_polar)

## Draw the coordinates via 3D projection, colored by the polar coordinate
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2], marker='o', c=B_polar)
```

![mobius band](https://github.com/peekxc/tallem/blob/main/resources/tallem_polar.png?raw=true)