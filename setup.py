
# All of these should be available at setup time due to pyproject.toml
import os
import sys
import re
import sysconfig
import platform
import subprocess

from glob import glob
# from skbuild import setup  # This line replaces 'from setuptools import setup'
from setuptools import setup, Extension, Command, Extension, find_packages
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile, naive_recompile

ext_modules = [
	Pybind11Extension(
		"python_example",
		["src/tallem/example.cpp", "src/tallem/carma_svd.cpp"],
		cxx_std=17
	),
]

# Convert distutils Windows platform specifiers to CMake -A arguments
PLATFORMS_TO_CMAKE = { "win32": "Win32", "win-amd64": "x64" }

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
		def __init__(self, name, sourcedir=""):
				Extension.__init__(self, name, sources=[])
				self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
	def run(self):
		try:
			out = subprocess.check_output(['cmake', '--version'])
		except OSError:
			raise RuntimeError("CMake must be installed to build the following extensions: ".join(e.name for e in self.extensions))
		if platform.system() == "Windows":
			cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
			if cmake_version < '3.1.0':
				raise RuntimeError("CMake >= 3.1.0 is required on Windows")
		for ext in self.extensions:
			self.build_extension(ext)

	def build_extension(self, ext):
		extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
		cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir, '-DPYTHON_EXECUTABLE=' + sys.executable]
		cfg = 'Debug' if self.debug else 'Release'
		build_args = ['--config', cfg]

		# Pile all .so in one place and use $ORIGIN as RPATH
		cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
		cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]

		if platform.system() == "Windows":
			cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
			if sys.maxsize > 2**32: cmake_args += ['-A', 'x64']
			build_args += ['--', '/m']
		else:
			cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
			build_args += ['--', '-j2']
			env = os.environ.copy()
			env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
				env.get('CXXFLAGS', ''),
				self.distribution.get_version())
			if not os.path.exists(self.build_temp):
				os.makedirs(self.build_temp)
			subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
			subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
			print()  # Add an empty line for cleaner output
		
		# extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

		# # required for auto-detection of auxiliary "native" libs
		# if not extdir.endswith(os.path.sep):
		# 	extdir += os.path.sep

		# cfg = "Debug" if self.debug else "Release"

		# # CMake lets you override the generator - we need to check this.
		# # Can be set with Conda-Build, for example.
		# cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

		# # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
		# # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
		# # from Python.
		# cmake_args = [
		# 		"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
		# 		"-DPYTHON_EXECUTABLE={}".format(sys.executable),
		# 		"-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
		# 		"-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
		# ]
		# build_args = []

		# if self.compiler.compiler_type != "msvc":
		# 	# Using Ninja-build since it a) is available as a wheel and b)
		# 	# multithreads automatically. MSVC would require all variables be
		# 	# exported for Ninja to pick it up, which is a little tricky to do.
		# 	# Users can override the generator with CMAKE_GENERATOR in CMake
		# 	# 3.15+.
		# 	if not cmake_generator:
		# 		cmake_args += ["-GNinja"]
		# else:
		# 	# Single config generators are handled "normally"
		# 	single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

		# 	# CMake allows an arch-in-generator style for backward compatibility
		# 	contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

		# 	# Specify the arch if using MSVC generator, but only if it doesn't contain a 
		# 	# backward-compatibility arch spec already in the generator name.
		# 	if not single_config and not contains_arch:
		# 		cmake_args += ["-A", PLATFORMS_TO_CMAKE[self.plat_name]]

		# 	# Multi-config generators have a different way to specify configs
		# 	if not single_config:
		# 		cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
		# 		build_args += ["--config", cfg]

		# # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
		# # across all generators.
		# if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
		# 	# self.parallel is a Python 3 only way to set parallel jobs by hand
		# 	# using -j in the build_ext call, not supported by pip or PyPA-build.
		# 	if hasattr(self, "parallel") and self.parallel:
		# 			# CMake 3.12+ only.
		# 			build_args += ["-j{}".format(self.parallel)]

		# if not os.path.exists(self.build_temp):
		# 	os.makedirs(self.build_temp)

		# subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
		# subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)
		# print()  # Add an empty line for cleaner output


# Add CMake as a build requirement if cmake is not installed or is too low a version
# from packaging.version import LegacyVersion
# from skbuild.exceptions import SKBuildError
# from skbuild.cmaker import get_cmake_version
# setup_requires = []
# try:
#     if LegacyVersion(get_cmake_version()) < LegacyVersion("3.4"):
#         setup_requires.append('cmake')
# except SKBuildError:
#     setup_requires.append('cmake')

# Optional multithreaded build
# ParallelCompile("0", needs_recompile=naive_recompile).install()

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
	author_email="matt.piekenbrock@gmail.com",
	url="https://github.com/pybind/python_example",
	description="A test project using pybind11",
	long_description="",
	#ext_modules=ext_modules, # required by pybind11
	ext_modules = [CMakeExtension("tallem/tallem")],
	cmdclass={"build_ext": CMakeBuild},
	zip_safe=False,
	packages=find_packages(where = 'src'),
	package_dir={"": "src"}
	# cmake_install_dir="src/cmake_build",
	# include_package_data = True,
	# cmake_args=['-DSOME_FEATURE:BOOL=OFF'],
	# cmake_install_dir="",
	# setup_requires=setup_requires
)