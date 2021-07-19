## TALLEM needs C++17 or higher 
project('tallem', 'cpp', default_options : ['cpp_std=c++17',])

## Find Python 3.9.1 or higher + PyBind11
py = import('python')
py_mod = py.find_installation('python3')
py_dep = py_mod.dependency(version : '>=3.9.1', components: ['Interpreter', 'Development', 'NumPy'])
py_b11 = dependency('pybind11')

numpy_include_dir = py_mod.get_path('purelib') / 'numpy' / 'core' / 'include'
message('Using NumPy includes from: ' + numpy_include_dir)

## Find armadillo
arma = dependency('armadillo')

## Includes
carma_include = include_directories('/usr/local/carma/include')
numpy_include = include_directories(numpy_include_dir)

## Concatenate everything 
includes = [numpy_include]
dependencies = [py_dep, py_b11] 

## Targets
src_dir = meson.current_source_dir()
install_dir = src_dir / 'src' / 'tallem'
py_mod.extension_module('carma_svd', sources : 'src/tallem/carma_svd.cpp', dependencies : dependencies + [arma], include_directories : includes + [carma_include], install : true, install_dir : install_dir)
py_mod.extension_module('fast_svd', sources : 'src/tallem/fast_svd.cpp', dependencies : dependencies, include_directories : includes, install : true, install_dir : install_dir)