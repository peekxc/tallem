
# All of these should be available at setup time due to pyproject.toml
import sys
from glob import glob
from skbuild import setup  # This line replaces 'from setuptools import setup'
# from setuptools import setup, Command
from setuptools import find_packages
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile, naive_recompile

ext_modules = [
	Pybind11Extension(
		"python_example",
		["src/example.cpp", "src/carma_svd.cpp"],
		cxx_std=17
	),
]

# Optional multithreaded build
# ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

# class CleanCommand(Command):
# 	"""Custom clean command to tidy up the project root."""
# 	user_options = []
# 	def initialize_options(self): pass
# 	def finalize_options(self): pass
# 	def run(self):
# 			os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(
	name="tallem",
	version="0.1.0",
	author="Matt Piekenbrock",
	author_email="sylvain.corlay@gmail.com",
	url="https://github.com/pybind/python_example",
	description="A test project using pybind11",
	long_description="",
	ext_modules=ext_modules, # required by pybind11
	extras_require={"test": "pytest"},
	cmdclass={
		"build_ext": build_ext
	},
	zip_safe=False,
	# packages=find_packages(where = 'src'),
	package_dir={"": "src"},
	cmake_install_dir="src/cmake_build",
	include_package_data = True,
	# cmake_args=['-DSOME_FEATURE:BOOL=OFF'],
	# cmake_install_dir=""
)