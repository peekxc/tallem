## TALLEM needs C++17 or higher 
project('tallem', 'cpp', 'fortran',
	version: '0.1.1',
  default_options : ['buildtype=release', 'warning_level=3', 'cpp_std=c++17']
)
project_dir = meson.current_source_dir()

## Folder variables 
fs = import('fs')
home_dir = fs.expanduser('~')

## Link with LAPACK
fc = meson.get_compiler('fortran')
lapack = dependency('lapack', cmake_module_path: 'cmake_modules', required: false)
if not lapack.found()
  lapack_proj = subproject('lapack')
  lapack = lapack_proj.get_variable('lapack')
endif

## Find Python 3.8+ for Protocol classes
py = import('python')
py_mod = py.find_installation('python3')
py_dep = py_mod.dependency(version : '>=3.8.0', components: ['Interpreter', 'Development', 'NumPy'])
py_b11 = dependency('pybind11')

## Need numpy includes explicitly
numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
message('Using NumPy includes from: ' + numpy_include_dir)

## Find armadillo
add_global_arguments('-Wno-non-virtual-dtor', language : 'cpp') # to remove installation warnings
# add_global_arguments('-Wdelete-non-virtual-dtor', language : 'cpp') 
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
message('Using CARMA link target from: ' + carma_inc)

## If not found, try searching all the system libraries for it, otherwise as a last resort
## attempt a subproject build  
if carma_inc == '.'
	carma = dependency('carma', required : true) # search for it via Cmake module 
endif 

## Include NumPy headers
numpy_include = include_directories(numpy_include_dir)

## Target(s)
add_global_arguments(['-llapack', '-lblas'], language : 'cpp')

py_mod.extension_module('fast_svd',
	sources : 'src/tallem/fast_svd.cpp', 
	dependencies : [py_dep, py_b11, arma, lapack], 
	include_directories : [numpy_include, carma_inc],
	install : true, 
	install_dir : project_dir / 'src' / 'tallem',
	link_args : '-llapack'
)

py_mod.extension_module('landmark', 
	sources : 'src/tallem/landmark.cpp', 
	dependencies : [py_dep, py_b11, arma, lapack], 
	include_directories : [numpy_include, carma_inc], 
	install : true, 
	install_dir : project_dir / 'src' / 'tallem'
)
