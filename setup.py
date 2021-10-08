## setup.py shim
import os
import sys
from setuptools import setup, find_packages

print("\n===== Running setuptools =====\n")
setup(
	## Use minimum amount of meta-data setuptools needs to not issue warnings
	name="tallem",
	version="0.1.1",
	maintainer="Matt Piekenbrock", 
	maintainer_email="matt.piekenbrock@gmail.com",
	url="https://github.com/peekxc/tallem",
	## Possible space optimization, but only relevent for eggs, not wheels. 
	zip_safe=False,
	## minimum python version (Protocol classes need 3.8+)
	python_requires=">=3.8",
	## package dependencies needed at run-time (but not at install-time)
	setup_requires=["numpy>=1.7.0", "scipy"], 
	## Package source modules---tell setuptools that nothing outside src/* should be added to global namespace 
	packages=find_packages(where='src'), package_dir={'':'src'}, 
	## Optional packages, grouped by [arbitrary] identifiers
	extras_require={
		"extras": ["scikit-learn", "autograd", "pymanopt"],
	}
)

# name="tallem",
# version="0.1.1",
# author="Matt Piekenbrock",
# author_email="matt.piekenbrock@gmail.com",
# url="https://github.com/peekxc/tallem",
# description="Topological Assembly of Local Euclidean Models",
# long_description="The project contains a python implementation of TALLEM, a non-linear dimensionality reduction method",

## Notes: 
## using 'Extension' module from distutils.core builds shared extension via setuptools after manually 
## specifying dependencies, e.g.
##
## ext_modules=[Extension(name='fast_svd', sources=['src/tallem/pbm/fast_svd.cpp', include_dirs=...)]
## 
## But building extension modules is already handled by meson.build, so this is not needed in setup.py
## It appears possible to call meson from within setup.py via 
## 
## cmdclass=dict(build_ext=...), # build command
## 
## However, this step would only decrease the number of commands to *build* the package wheel. This may or may not be
## needed if a wheel install fails and the user has to fallback to building from the source distribution. 
## TODO: consider poetry to automate this more easily, see https://stackoverflow.com/questions/63326840/specifying-command-line-scripts-in-pyproject-toml