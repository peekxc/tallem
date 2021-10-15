## TALLEM needs C++17 or higher 
project('tallem', 'cpp', 'fortran',
	version: '0.1.1',
  default_options : ['buildtype=release', 'warning_level=3', 'cpp_std=c++17']
)
project_dir = meson.current_source_dir()
# add_project_arguments('-Db_pie=true -Db_staticpic=true', language : 'cpp')
add_project_arguments('-Wno-non-virtual-dtor', language : 'cpp') # to remove installation warnings
add_project_arguments('-DCARMA_DONT_REQUIRE_F_CONTIGUOUS=OFF', language : 'cpp')

## Folder variables 
fs = import('fs')
home_dir = fs.expanduser('~')

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
py_mod = py.find_installation('python3', modules : ['numpy'], disabler:true, required:false)
py_dep = py_mod.dependency(version : '>=3.8.0', components: ['Interpreter', 'Development', 'NumPy'])
# py_b11 = dependency('pybind11')

## Detect numpy
numpy = run_command(py_mod,'-c','import numpy;print(numpy.get_include())')
if numpy.returncode()==0
	message ('Numpy found: '+ numpy.stdout())
	pybind_inc = include_directories(numpy.stdout().strip())
else
	message ('Numpy not found: building without arrays')
	pybind_inc = include_directories()
endif


# np = py_mod.get_path('numpy')
# print(np)

## Need numpy includes explicitly
numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
# np = import('numpy')
# py_mod.get_path('numpy')
message('Detected numpy include from: ' + pybind_inc)
message('Using NumPy includes from: ' + numpy_include_dir)


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
numpy_include = include_directories(pybind_inc)
pybind11_include = include_directories('.' / 'extern')

## Target(s)
py_mod.extension_module('fast_svd',
	sources : project_dir / 'src' / 'tallem' / 'pbm' / 'fast_svd.cpp', 
	dependencies : [py_dep, arma], 
	include_directories : [numpy_include, carma_includes, pybind11_include],
	install : true, 
	install_dir : project_dir / 'src' / 'tallem' / 'pbm'
)

py_mod.extension_module('landmark', 
	sources :  project_dir / 'src' / 'tallem' / 'pbm' / 'landmark.cpp', 
	dependencies : [py_dep, arma], 
	include_directories : [numpy_include, carma_includes, pybind11_include], 
	install : true, 
	install_dir : project_dir / 'src' / 'tallem' / 'pbm'
)
