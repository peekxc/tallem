## TALLEM needs C++17 or higher 
project('tallem', 'cpp', default_options : ['cpp_std=c++17'])
project_dir = meson.current_source_dir()
message(project_dir)

## Folder variables 
fs = import('fs')
home_dir = fs.expanduser('~')

## To remove armadillo installation warnings
add_global_arguments('-Wnon-virtual-dtor', language : 'cpp')
add_global_arguments('-llapack', language : 'cpp')
add_global_arguments('-lblas', language : 'cpp')

## Find Python 3.9.1 or higher + PyBind11
py = import('python')
py_mod = py.find_installation('python3')
py_dep = py_mod.dependency(version : '>=3.9.1', components: ['Interpreter', 'Development', 'NumPy'])
py_b11 = dependency('pybind11')

## Need numpy includes explicitly
numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
message('Using NumPy includes from: ' + numpy_include_dir)

## Find LAPACK/BLAS via FindLAPACK and FindBLAS cmake modules 
lapack = dependency('lapack', required: true, method: 'cmake')
blas = dependency('blas', required: true, method: 'cmake')

## Find armadillo
arma = dependency('armadillo', version : '>=10.5.2', required : true)

## Attempt to find CARMA intelligently
add_project_arguments('-DCARMA_DONT_REQUIRE_F_CONTIGUOUS=OFF', language : 'cpp')
carma_dirs = [ 
	'usr' / 'local' / 'include' / 'carma',
	'usr' / 'local' / 'carma' / 'include',
	project_dir / 'carma' / 'include',
	home_dir / 'carma' / 'include',
		'usr' / 'local' / 'include'
]
carma_inc = '.'
foreach carma_path : carma_dirs
	if fs.is_dir(carma_path)
		carma_inc = carma_path
		break
	endif
endforeach
message(carma_inc)

## If not found, try searching all the system libraries for it, otherwise as a last resort
## attempt a subproject build  
if carma_inc == '.'
	carma = dependency('carma', required : true) # search for it via Cmake module 
endif 

## Include NumPy headers
numpy_include = include_directories(numpy_include_dir)

## Target(s)
py_mod.extension_module('fast_svd',
	sources : 'src/tallem/fast_svd.cpp', 
	dependencies : [py_dep, py_b11, arma, blas], 
	include_directories : [numpy_include, carma_inc],
	install : true, 
	install_dir : project_dir / 'src' / 'tallem'
)

py_mod.extension_module('landmark', 
	sources : 'src/tallem/landmark.cpp', 
	dependencies : [py_dep, py_b11, arma], 
	include_directories : [numpy_include, carma_inc], 
	install : true, 
	install_dir : project_dir / 'src' / 'tallem'
)
