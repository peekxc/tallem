## TALLEM needs C++17 or higher 
project('tallem', 'cpp', default_options : ['cpp_std=c++17'])
project_dir = meson.current_source_dir()
message(project_dir)
## Folder variables 
fs = import('fs')
home_dir = fs.expanduser('~')

## To remove armadillo installation warnings
add_global_arguments('-Wnon-virtual-dtor', language : 'cpp')

## Find Python 3.9.1 or higher + PyBind11
py = import('python')
py_mod = py.find_installation('python3')
py_dep = py_mod.dependency(version : '>=3.9.1', components: ['Interpreter', 'Development', 'NumPy'])
py_b11 = dependency('pybind11')

## Need numpy includes explicitly
numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
message('Using NumPy includes from: ' + numpy_include_dir)

## Find armadillo
arma = dependency('armadillo', version : '>=10.5.2', required : true)

## Attempt to find CARMA intelligently
add_project_arguments('-DCARMA_DONT_REQUIRE_F_CONTIGUOUS=OFF', language : 'cpp')
carma_dirs = [ 
	'usr' / 'local' / 'carma' / 'include',
	home_dir / 'carma' / 'include'
]
carma_inc = '.'
foreach carma_path : carma_dirs
	if fs.is_dir(carma_path)
		carma_inc = carma_path
		carma = declare_dependency(include_directories : carma_inc)
		break
	endif
endforeach

## If not found, try searching all the system libraries for it, otherwise as a last resort
## attempt a subproject build  
if carma_inc == '.'
	carma = dependency('carma', required : false)
	if not(carma.found())
		cmake = import('cmake')
		carma_subproj = cmake.subproject('carma')
		carma = carma_subproj.dependency('carma')
	endif
endif 

## Include NumPy headers
numpy_include = include_directories(numpy_include_dir)

## Target(s)
py_mod.extension_module('fast_svd', 
	sources : 'src/tallem/fast_svd.cpp', 
	dependencies : [py_dep, py_b11, arma, carma], 
	include_directories : [numpy_include], 
	install : true, 
	install_dir : project_dir / 'src' / 'tallem'
)

py_mod.extension_module('landmark', 
	sources : 'src/tallem/landmark.cpp', 
	dependencies : [py_dep, py_b11, arma, carma], 
	include_directories : [numpy_include], 
	install : true, 
	install_dir : project_dir / 'src' / 'tallem'
)
