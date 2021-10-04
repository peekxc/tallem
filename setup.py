# All of these should be available at setup time due to pyproject.toml
import os
import sys
from setuptools import setup, find_packages

setup(
	name="tallem",
	version="0.1.0",
	author="Matt Piekenbrock",
	author_email="matt.piekenbrock@gmail.com",
	url="https://github.com/peekxc/tallem",
	description="Topological Assembly of Local Euclidean Models",
	long_description="The project contains a python implementation of TALLEM, a non-linear dimensionality reduction method",
	zip_safe=False,
	packages=find_packages(where = 'src'),
	package_dir={"": "src"},
	py_modules=['fast_svd', 'landmark']
	classifiers=[
		"Development Status :: 3 - Alpha", 
		"License :: OSI Approved :: Apache Software License", 
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Intended Audience :: Science/Research"
	]
)