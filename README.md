# Topological Assembly of Local Euclidean Models (TALLEM)

TALLEM is a topologically inspired non-linear dimensionality reduction method. Given some data set _X_ and a map 
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=f%20%3A%20X%20%5Cto%20B"></div>
onto some topological space _B_ which captures both the topology and nonlinearity of _X_, TALLEM constructs a map 
<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20"></div>
mapping _X_ to a lower-dimensional space. 

This repository hosts `tallem`, a [Python project](https://packaging.python.org/glossary/#term-Project) which implements TALLEM. Currently, the `tallem`
package must be built from source, i.e. no wheels are available on PyPI yet. 

## Dependencies 

`tallem` requires _Python >= 3.9.1_, along with an assortment of packages listed in `pyproject.toml`. This are automatically downloaded and installed using the installation procedure outlined below.

Outside of Python, `tallem` uses [pybind11](https://github.com/pybind/pybind11/tree/stable) to interface with a variety of software libraries using [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), 
which themselves must be installed in order to run TALLEM. These include: 

* [Armadillo](http://arma.sourceforge.net/) >= 10.5.2
* [CARMA](https://github.com/RUrlus/carma) >= v0.5
* [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/) (for building the [extension modules](https://docs.python.org/3/glossary.html#term-extension-module))

Since prebuilt wheels are not yet provided, a C++17-compliant compiler will be needed to install `tallem`. 

## Installing

Meson and Ninja are installeable with pip:

```bash
pip install meson ninja 
```

Armadillo [provides a variety of installation options](http://arma.sourceforge.net/download.html), including pre-built binaries available for download or via system package managers like Homebrew. 

Though header-only, CARMA requires building from source using [CMAKE](https://cmake.org/runningcmake/). On UNIX-like systems, this can be achieved via: 

```bash
git clone https://github.com/RUrlus/carma
cd carma | cmake . | make | sudo make install 
```

Ensure the path to CARMA in the `meson.build` script matches where it was installed (e.g. `/usr/local/carma/include`). 

`tallem` is [PEP 517/518](https://www.python.org/dev/peps/pep-0518/) compliant: a distribution can be built using [`build`](https://pypa-build.readthedocs.io/en/stable/):

> python -m mesonbuild.mesonmain build
> meson install -C build
> python -m build 

Assuming this succeeds, the package distribution should be located in the `dist` folder, from which it can be installed the the local [site-packages](https://docs.python.org/3/library/site.html#site.USER_SITE) via: 

> pip install dist/tallem-*.whl
