import os 
import sys 
import glob
import shutil
from pathlib import Path
import mesonbuild

# From: https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
def expandpath(path_pattern):
	p = Path(path_pattern).expanduser()
	parts = p.parts[p.is_absolute():]
	return Path(p.root).glob(str(Path(*parts)))

def build(setup_kwargs):
	suffix = os.popen('python3-config --extension-suffix').read().rstrip()
	print(f"Building extensions with suffix: {suffix}")
	home_dir = os.getcwd()
	existing_modules = list(expandpath(f"{home_dir}/src/tallem/pbm/*{suffix}"))
	if len(existing_modules) > 0:
		print("Removing existing modules for a clean build")
	## Remove existing extension modules
	for m in existing_modules:
		os.remove(m)
	
	import numpy as np
	print("\n==== NUMPY INCLUDES ====\n")
	print(f"{np.get_include()}")

	# Use cython -a *.pyx
	import Cython.Compiler.Options
	Cython.Compiler.Options.annotate = True

	from Cython.Compiler.Options import get_directive_defaults
	directive_defaults = get_directive_defaults()
	directive_defaults['linetrace'] = True
	directive_defaults['binding'] = True

	# print("\n==== CYTHONIZING *.pyx files ====\n")
	# from Cython.Build import cythonize
	# from setuptools import Extension
	# from setuptools.dist import Distribution
	# from distutils.command.build_ext import build_ext

	# extensions = [Extension(
	# 	name='mds_cython',
	# 	sources=[f"{home_dir}/src/tallem/mds_cython.pyx"],
	# 	extra_compile_args=[''],
	# 	extra_link_args=['-fopenmp', '-I/usr/local/Cellar/libomp/12.0.1/include', '-lomp', '-L/usr/local/Cellar/libomp/12.0.1/lib']
	# )]
	# setup_kwargs.update({
	# 	'ext_modules': cythonize(
	# 		module_list=extensions,
	# 		aliases={ 'NUMPY_INCLUDE': np.get_include() },
	# 		include_path=[np.get_include()],
	# 		language_level=3,
	# 		compiler_directives={'linetrace': True},
	# 		annotate=True
	# 	),
	# 	'cmdclass': {'build_ext': build_ext}
	# })

	## Recompile
	print("\n==== Printing compiler version ====\n")
	os.system("c++ --version")
	print("\n==== Starting meson build ====\n")
	os.system("python3 -m mesonbuild.mesonmain build")
	# os.system("meson setup build")
	# os.system("meson compile -vC build")
	target_path = next(expandpath(f"{home_dir}/src/tallem/extensions/")).resolve()
	print(f"\n==== Extension module install path: {target_path} ====\n")
	for file in glob.glob(f"build/*{suffix}"):
		shutil.copy(file, target_path)
	# os.system(f"cp build/*{suffix} {target_path}")

	print("\n==== Finished meson build ====\n")
	
	# os.system("meson install -C build")

	## Check if they now exist
	num_so = len([p.name for p in expandpath(f"{home_dir}/src/tallem/extensions/*{suffix}")])
	if num_so > 0:
		return(0)
	else: 
		print("Did not detect native python extensions; Exiting build")
		sys.exit(-1)
