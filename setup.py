# -*- coding: utf-8 -*-
import os 
import sys 
import pathlib
import importlib
import glob
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import mesonbuild
import platform

suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

package_dir = \
{'': 'src'}

packages = \
['tallem', 'tallem.extensions']

package_data = \
{'': ['*'], 'tallem.extensions': ['*.so', '*.pyd', 'extensions/*.so', 'extensions/*.pyd'] }

install_requires = \
['numpy>=1.21.3,<2.0.0', 'scipy>=1.6']

extras_require = \
{'autograd': ['autograd'],
 'pymanopt': ['pymanopt>=0.2.5'],
 'scikit-learn': ['scikit-learn>=1.0']}

# From: https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
def expandpath(path_pattern):
	p = Path(path_pattern).expanduser()
	parts = p.parts[p.is_absolute():]
	return Path(p.root).glob(str(Path(*parts)))

def build_extensions(setup_kwargs):
	print(f"Building extensions with suffix: {suffix}")
	home_dir = os.getcwd()
	existing_modules = list(expandpath(f"{home_dir}/src/tallem/extensions/*{suffix}"))
	if len(existing_modules) > 0:
		print("Removing existing modules for a clean build")

	## Remove existing extension modules
	for m in existing_modules: os.remove(m)
	
	import numpy as np
	print("\n==== NUMPY INCLUDES ====\n")
	print(f"{np.get_include()}")

	## Recompile
	print("\n==== Printing compiler version ====\n")
	os.system("c++ --version")
	
	## Check if build exists, and if it does remove it
	if os.path.isdir(f"{home_dir}/build"):
		print(f"\n==== Removing directory {home_dir}/build ====\n")
		shutil.rmtree(f"{home_dir}/build")

	print("\n==== Starting meson build ====\n")
	os.system("python3 -m mesonbuild.mesonmain build")
	os.system("python3 -m mesonbuild.mesonmain compile -vC build")
	
	## Linux CI servers raise tty exception on meson install, so do manual copy instead
	os.system("python3 -m mesonbuild.mesonmain install -C build")
	target_path = next(expandpath(f"{home_dir}/src/tallem/extensions/")).resolve()
	print(f"\n==== Extension module install path: {target_path} ====\n")
	for file in glob.glob(f"build/*{suffix}"):
		print(f"Installing {file} to: {target_path} \n")
		shutil.copy(file, target_path)

	print("\n==== Finished meson build ====\n")

	## Check if they now exist
	num_so = len([p.name for p in expandpath(f"{home_dir}/src/tallem/extensions/*{suffix}")])
	if num_so > 0:
		return(0)
	else: 
		print("ERROR: Did not detect native python extensions; Exiting build")
		sys.exit(-1)

# Boilerplate from https://stackoverflow.com/questions/63350376/place-pre-compiled-extensions-in-root-folder-of-non-pure-python-wheel-package
# because setuptools/distutils are archaic tools  
# class CustomDistribution(Distribution):
#   def iter_distribution_names(self):
#     for pkg in self.packages or ():
#       yield pkg
#     for module in self.py_modules or ():
#       yield module

class CustomExtension(Extension):
  def __init__(self, path):
    self.path = path
    super().__init__(pathlib.PurePath(path).name, [])

class build_CustomExtensions(build_ext):
  def run(self):
    for ext in (x for x in self.extensions if isinstance(x, CustomExtension)):
      source = f"{ext.path}{suffix}"
      build_dir = pathlib.PurePath(self.get_ext_fullpath(ext.name)).parent
      os.makedirs(f"{build_dir}/{pathlib.PurePath(ext.path).parent}", exist_ok = True)
      shutil.copy(f"{source}", f"{build_dir}/{source}")

def find_extensions(directory):
  extensions = []
  for path, _, filenames in os.walk(directory):
    for filename in filenames:
      filename = pathlib.PurePath(filename)
      if pathlib.PurePath(filename).suffix == suffix:
        extensions.append(CustomExtension(os.path.join(path, filename.stem)))
  return extensions

setup_kwargs = {
    'name': 'tallem',
    'version': '0.2.2',
    'description': 'Topological Assembly of Locally Euclidean Models',
    'long_description': '# Topological Assembly of Local Euclidean Models \n\nThis repository implements TALLEM - a topologically inspired non-linear dimensionality reduction method.\n\nGiven some data set *X* and a map <img class=\'latex-inline math\' style="background: white; vertical-align:-0.105206pt;" src="https://render.githubusercontent.com/render/math?math=\\large f%20%3A%20X%20%5Cto%20B&mode=inline"> onto some topological space _B_ which captures the topology/nonlinearity of _X_, TALLEM constructs a map <img style="background: white; vertical-align:-0.105206pt" class=\'latex-inline math\' src="https://render.githubusercontent.com/render/math?math=\\large F%20%3A%20X%20%5Cto%20%5Cmathbb%7BR%7D%5ED%20&mode=inline"> mapping _X_ to a _D_-dimensional space. \n\nTODO: describe TALLEM more\n\n## Installing + Dependencies \n\n`tallem`\'s run-time dependencies are fairly minimal. They include:  \n\n1. _Python >= 3.8.0_ \n2. *NumPy (>= 1.20)* and *SciPy* *(>=1.6)*\n\nThe details of the rest of package requirements are listed in [pyproject.toml](https://github.com/peekxc/tallem/blob/main/pyproject.toml). These are automatically downloaded and installed via `pip`: \n\n\n\nSome functions which extend TALLEM\'s core functionality require additional dependencies to be called---they include *autograd*, *pymanopt*, *scikit-learn*, and *bokeh*. These  packages are completely optional, i.e. they are not needed to get the resulting embedding. Nonetheless, if you would like these package as well, use: \n\n\n\n\n\n###Installing from cibuildwheels\n\nTODO\n\n### Installing from source\n\nTo install `tallem` from source, clone the repository and install the package via: \n\n```bash\npython -m pip install .\n```\n\n`tallem` relies on a few package dependencies in order to compile correctly when building from source. These libraries include: \n\n* [Armadillo](http://arma.sourceforge.net/) >= 10.5.2 ([see here for installation options](http://arma.sourceforge.net/download.html))\n* [Poetry](https://python-poetry.org/) (for building the [source](https://packaging.python.org/glossary/#term-Source-Distribution-or-sdist) and [binary](https://packaging.python.org/glossary/#term-Wheel) distributions)\n* [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/) (for building the [extension modules](https://docs.python.org/3/glossary.html#term-extension-module))\n\nAn install attempt of these external dependencies is made if they are not available prior to call to `pip`, however these may require manual installation. Additionally, the current source files are written in [C++17](https://en.wikipedia.org/wiki/C%2B%2B17), so a [C++17 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support/17) will be needed. If you have an installation problems or questions, feel free to [make a new issue](https://github.com/peekxc/tallem/issues).\n\n## Usage \n\nBelow is some example code showcasing TALLEMs ability to handle topological obstructions to dimensionality reduction like non-orientability.  \n\n```python\nfrom tallem import TALLEM\nfrom tallem.cover import IntervalCover\nfrom tallem.datasets import mobius_band\n\n## Get mobius band data + its parameter space\nX, B = mobius_band()\nB_polar = B[:,[1]]\n\n## Construct a cover over the polar coordinate\nm_dist = lambda x,y: np.sum(np.minimum(abs(x - y), (2*np.pi) - abs(x - y)))\ncover = IntervalCover(B_polar, n_sets = 10, overlap = 0.30, metric = m_dist)\n\n## Parameterize TALLEM + transform the data to the obtain the coordinization\nemb = TALLEM(cover=cover, local_map="cmds2", n_components=3).fit_transform(X, B_polar)\n\n## Draw the coordinates via 3D projection, colored by the polar coordinate\nimport matplotlib.pyplot as plt\nfig = plt.figure()\nax = fig.add_subplot(projection=\'3d\')\nax.scatter(*emb.T, marker=\'o\', c=B_polar)\n```\n\n![mobius band](https://github.com/peekxc/tallem/blob/main/resources/tallem_polar.png?raw=true)\n\n',
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
    'python_requires': '>=3.8,<3.10',
    'ext_modules': find_extensions("src/tallem"),
    'cmdclass': {'build_ext': build_CustomExtensions},
    # 'distclass': CustomDistribution
}

# Build first, then invoke setup 
build_extensions(setup_kwargs)
setup(**setup_kwargs)

