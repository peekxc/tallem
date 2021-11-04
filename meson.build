## TALLEM needs C++17 or higher 
project('tallem', 'cpp', 'fortran', 'cython',
	version: '0.1.1',
  default_options : [
		'cpp_std=c++17', 'warning_level=3',
		# 'buildtype=debug', 'optimization=0', 'debug=true', 
		'buildtype=release', 'optimization=2', 'debug=false',
		'b_lto_mode=default', 'b_pie=true', 'b_sanitize=none',
		'install_umask=002', 'strip=false'
	]
)
# For release: use -02 -fPIC 
# For benchmarking: consider -03 -march=native
project_dir = meson.current_source_dir()
# add_project_arguments('-stdlib=libc++', language: 'c')
# add_project_arguments('-stdlib=libc++', language: 'cpp')
add_project_arguments('-Wno-non-virtual-dtor', language : 'cpp') # to remove installation warnings
# add_project_arguments('-DCARMA_DONT_REQUIRE_F_CONTIGUOUS=OFF', language : 'cpp')

## Check the C++ compiler location
compiler = meson.get_compiler('cpp')
message(compiler.get_id())

## Folder variables 
fs = import('fs')
home_dir = fs.expanduser('~')

# openmp = dependency('openmp', required : false, language : 'cpp')
# if openmp.found()
# 	#_c = compiler.has_argument('-fopenmp')
#   #_h = compiler.has_header('omp.h')
# 	message('using OpenMP')
# endif

## Link with LAPACK
# fc = meson.get_compiler('fortran')
# lapack = dependency('lapack', cmake_module_path: 'cmake_modules', required: false)
# if not lapack.found()
# 	message('LAPACK not found. Attempting to build from subproject.')
#   lapack_proj = subproject('lapack')
#   lapack = lapack_proj.get_variable('lapack')
# endif

## Query the variables to see what we're working with
# lapack_lib = lapack.get_variable(cmake: 'LAPACK_LIBRARIES', default_value: '')
# lapack_linker_flags = lapack.get_variable(cmake: 'LAPACK_LINKER_FLAGS', default_value : ['-llapack', '-lblas'])
# message('LAPACK libraries found: ', lapack_lib)
# message('LAPACK linker flags: ', lapack_linker_flags)

## Find Python 3.8+ for Protocol classes
py = import('python')
py_mod = py.find_installation('python3', modules:['numpy'], disabler:true, required:false)
py_dep = py_mod.dependency(version:'>=3.8.0', components:['Interpreter', 'Development', 'NumPy'])

## Detect numpy
numpy = run_command(py_mod,'-c','import numpy;print(numpy.get_include())')
if numpy.returncode()==0
	message ('Numpy found: '+ numpy.stdout())
	numpy_include = include_directories(numpy.stdout().strip())
else
	numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
	message('Using NumPy includes from: ' + numpy_include_dir)
	numpy_include = include_directories(numpy_include_dir)
endif


# np = py_mod.get_path('numpy')
# print(np)

## Need numpy includes explicitly
# numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
# np = import('numpy')
# py_mod.get_path('numpy')
# message('Detected numpy include from: ' + pybind_inc)

## Find armadillo, and if not found install it via subproject wrap
arma = dependency('armadillo', version: '>=10.5.2', required: false)
if not(arma.found())
	cmake = import('cmake')
	opt_var = cmake.subproject_options()
	opt_var.add_cmake_defines({'BUILD_SHARED_LIBS': true}) # Call CMake with `-DVAR=VALUE`
	arma_proj = cmake.subproject('arma', options: opt_var)
	arma = arma_proj.dependency('armadillo')
endif

## Attempt to find CARMA intelligently in the probable system library locations...
# carma_dirs = [ 
# 	'/usr' / 'local' / 'include' / 'carma',
# 	'/usr' / 'local' / 'carma' / 'include',
# 	'/usr' / 'local' / 'include'
# ]

# ## ...otherwise rely on the default git submodule 
# carma_inc = '.' / 'extern' / 'carma' / 'include'
# foreach carma_path : carma_dirs
# 	if fs.is_dir(carma_path) and fs.is_file(carma_path / 'carma')
# 		carma_inc = carma_path
# 		break
# 	endif
# endforeach
# message('Using CARMA link target from: ' + carma_inc)
carma_includes = include_directories('.' / 'extern' / 'carma' / 'include')

## Finally, if not found, attempt a subproject git pull and build  
# if not fs.is_dir(carma_inc)
# 	cmake = import('cmake')
# 	carma_proj = cmake.subproject('carma')
# 	carma = carma_proj.dependency('carma')
# endif 

## Include NumPy headers
# numpy_include = include_directories(numpy_include_dir)
# numpy_include = include_directories(pybind_inc)
pybind11_include = include_directories('.' / 'extern')

## Target(s)
py_mod.extension_module('fast_svd',
	sources : project_dir / 'src' / 'tallem' / 'extensions' / 'fast_svd.cpp', 
	dependencies : [py_dep, arma], 
	include_directories : [numpy_include, carma_includes, pybind11_include],
	install : true, 
	install_dir : project_dir / 'src' / 'tallem' / 'extensions'
)

py_mod.extension_module('landmark', 
	sources :  project_dir / 'src' / 'tallem' / 'extensions' / 'landmark.cpp', 
	dependencies : [py_dep, arma], 
	include_directories : [numpy_include, carma_includes, pybind11_include], 
	install : true, 
	install_dir : project_dir / 'src' / 'tallem' / 'extensions'
)

cython = find_program('cython')
mds_cython_cpp = custom_target('mds_cython',
  output : 'mds_cython.cpp',
  input : 'src' / 'tallem' / 'extensions' / 'mds_cython.pyx',
  command : [	
		cython, '@INPUT@', 
		'--cplus', '-3', '--line-directives', '-X', 'profile=True', '-X', 'linetrace=True', '-X', 'binding=True', '--annotate', # for debugging/profiling only
		# '--cplus', '-3', '-Wextra', # for release mode
		'-o', '@OUTPUT@'
	]
)

py_mod.extension_module('mds_cython',
	sources: mds_cython_cpp,
	dependencies : [py_dep],
	include_directories : [numpy_include],
	install_dir : project_dir / 'src' / 'tallem' / 'extensions',
	install : true,
	override_options : ['cython_language=cpp'], 
	cpp_args: '-DCYTHON_TRACE_NOGIL=1'
)
