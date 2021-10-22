# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['tallem', 'tallem.pbm']

package_data = \
{'': ['*']}

install_requires = \
['Shapely>=1.7,<2.0',
 'build>=0.7,<0.8',
 'meson>=0.59,<0.60',
 'ninja>=1.10,<2.0',
 'numpy>=1.1,<2.0',
 'pybind11>=2.8,<3.0',
 'scipy>=1.6,<2.0',
 'wheel>=0.37,<0.38']

extras_require = \
{'autograd': ['autograd>=1.3,<2.0'], 'pymanopt': ['pymanopt>=0.2.5,<0.3.0']}

setup_kwargs = {
    'name': 'tallem',
    'version': '0.1.3',
    'description': 'Topological Assembly of Locally Euclidean Models',
    'long_description': '# Topological Assembly of Local Euclidean Models \n\nThis repository implements TALLEM - a topologically inspired non-linear dimensionality reduction method.\n\nGiven some data set *X* and a map <img class=\'latex-inline math\' style="background: white; vertical-align:-0.105206pt;" src="https://render.githubusercontent.com/render/math?math=\\large f%20%3A%20X%20%5Cto%20B&mode=inline"> onto some topological space _B_ which captures the topology/nonlinearity of _X_, TALLEM constructs a map <img style="background: white; vertical-align:-0.105206pt" class=\'latex-inline math\' src="https://render.githubusercontent.com/render/math?math=\\large F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20&mode=inline"> mapping _X_ to a _D_-dimensional space. \n\nTODO: describe TALLEM more\n\n## Dependencies \n\n`tallem` requires _Python >= 3.9.1_, along with the packages listed in [pyproject.toml](https://github.com/peekxc/tallem/blob/a1e7d2cd5d0dab5816ece658a3816dc0425f2391/pyproject.toml#L12). These are automatically downloaded and installed via `pip` using the installation procedure given below.\n\nExternally, `tallem` uses [pybind11](https://github.com/pybind/pybind11/tree/stable) to interface with a variety of software libraries using [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), which themselves must be installed in order to run TALLEM. These include: \n\n* [Armadillo](http://arma.sourceforge.net/) >= 10.5.2\n* [CARMA](https://github.com/RUrlus/carma) >= v0.5\n* [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/) (for building the [extension modules](https://docs.python.org/3/glossary.html#term-extension-module))\n\nSince prebuilt wheels are not yet provided, a [C++17 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support/17) may be needed to install these dependencies. \n\n## Installing\n\nCurrently, `tallem` must be built from source--wheels will be made available on PyPI or some other host in the future. \n\nMeson and Ninja are installeable with `pip`:\n\n```bash\npip install meson ninja \n```\n\nArmadillo [provides a variety of installation options](http://arma.sourceforge.net/download.html).\n\nCARMA is a [header-only](https://en.wikipedia.org/wiki/Header-only), the source files only require the directory where the files are requires building from source using [CMAKE](https://cmake.org/runningcmake/). On UNIX-like terminals, this can be achieved via: \n\n```bash\ngit clone https://github.com/RUrlus/carma\ncd carma | cmake . | make | sudo make install \n```\n\nEnsure the path to CARMA in the `meson.build` script matches where it was installed (e.g. `/usr/local/carma/include`). \n\n`tallem` can be built using [`build`](https://pypa-build.readthedocs.io/en/stable/) package builder:\n\n```bash\npython -m mesonbuild.mesonmain build\nmeson install -C build\npython -m build \n```\n\nAssuming this succeeds, the [wheel](https://packaging.python.org/glossary/#term-Wheel) should be located in the `dist` folder, from which it can be installed the local [site-packages](https://docs.python.org/3/library/site.html#site.USER_SITE) via: \n\n```bash\npip install dist/tallem-*.whl\n```\n\nIf you have an installation problems or questions, feel free to [make a new issue](https://github.com/peekxc/tallem/issues).\n\n## Usage \n\nBelow is some example code showcasing TALLEMs ability to handle topological obstructions to dimensionality reduction like non-orientability.  \n\n```python\nfrom tallem import TALLEM\nfrom tallem.cover import IntervalCover\nfrom tallem.datasets import mobius_band\n\n## Get mobius band data + its parameter space\nX, B = mobius_band(n_polar=26, n_wide=6, embed=3).values()\nB_polar, B_radius = B[:,[1]], B[:,[0]]\n\n## Construct a cover over the polar coordinate\nm_dist = lambda x,y: np.sum(np.minimum(abs(x - y), (2*np.pi) - abs(x - y)))\ncover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, metric = m_dist)\n\n## Parameterize TALLEM + transform the data to the obtain the coordinization\nemb = TALLEM(cover=cover, local_map="cmds2", n_components=3).fit_transform(X, B_polar)\n\n## Draw the coordinates via 3D projection, colored by the polar coordinate\nimport matplotlib.pyplot as plt\nfig = plt.figure()\nax = fig.add_subplot(projection=\'3d\')\nax.scatter(*emb.T, marker=\'o\', c=B_polar)\n```\n\n![mobius band](https://github.com/peekxc/tallem/blob/main/resources/tallem_polar.png?raw=true)\n\n**FAQ**\n\n\n\n_The dependencies listed require Python 3.5+, but I\'m using an older version of Python. Will`tallem` still run on my machine, and if not, how can I make `tallem` compatible?_\n\n`tallem` requires Python version 3.5 or higher and will not run on older versions of Python. If your version of Python is older than this, consider installing `tallem` in a [virtual environment] that supports Python 3.5+. \nAlternatively, you\'re free to make the appropriate changes to `tallem` to make the library compatible with an older version yourself and then issue a PR. \n\n',
    'author': 'Matt Piekenbrock',
    'author_email': 'matt.piekenbrock@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/peekxc/tallem',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.8,<4.0'
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
