cmake_minimum_required(VERSION 3.15...3.19)
project(tallem)


#set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++17 -shared -fPIC")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-undefined,dynamic_lookup")
message("C++ compilation flags: ${CMAKE_CXX_FLAGS}")
set(SOURCE_DIR "src/tallem")

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/resources/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/resources/")


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/tallem)

FIND_PACKAGE( pybind11 REQUIRED )
FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs   3)


# IF(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND PYBIND11_FOUND )
#   INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS})

#   ADD_LIBRARY( example SHARED example.cc )

#   # The library must not have any prefix and should be located in a subfolder that includes the package name.  
#   SET_TARGET_PROPERTIES( example
#     PROPERTIES
#       PREFIX ""
#       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/tallem"
#   )

#   # This is required for linking the library under Mac OS X. Moreover,
#   # the suffix ensures that the module can be found by the interpreter
#   # later on.
#   IF( APPLE )
#     SET_TARGET_PROPERTIES( example
#       PROPERTIES
#         LINK_FLAGS "-undefined dynamic_lookup"
#         SUFFIX     ".so"
#     )
#   ENDIF()

#   # Place the initialization file in the output directory for the Python
#   # bindings. This will simplify the installation.
#   CONFIGURE_FILE( example/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/example/__init__.py )

#   # Ditto for the setup file.
#   CONFIGURE_FILE( example/setup.py ${CMAKE_CURRENT_BINARY_DIR}/example/setup.py )
# ENDIF()

# macro(remove_flag_from_target _target _flag)
#     get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
#     if(_target_cxx_flags)
#         list(REMOVE_ITEM _target_cxx_flags ${_flag})
#         set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
#     endif()
# endmacro()

# set(CMAKE_C_COMPILER "/usr/local/Cellar/llvm/12.0.0_1/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/12.0.0_1/bin/clang++")

# set(CMAKE_C_COMPILER "/usr/local/Cellar/gcc/11.1.0_1/bin/gcc-11")
# set(CMAKE_CXX_COMPILER "/usr/local/Cellar/gcc/11.1.0_1/bin/g++-11")

if(SKBUILD)
	message(STATUS "The project is built using scikit-build")
  # Scikit-Build does not add your site-packages to the search path automatically,
  # so we need to add it _or_ the pybind11 specific directory here.
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()


set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection") 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
message("C++ compilation flags: ${CMAKE_CXX_FLAGS}")




## Required CMake modules

find_package(Python3 3.9 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
# find_package(LAPACK REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++17)


# ---- CARMA TARGET ----- 
# set(USE_ARMA_VERSION 10.5.2)
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ /usr/local/Cellar/ REQUIRED)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/carma)
pybind11_add_module(carma_svd MODULE src/tallem/carma_svd.cpp)
target_include_directories(carma_svd PUBLIC ${PYTHON_INCLUDE_DIRS} ${CARMA_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(carma_svd PUBLIC ${ARMADILLO_LIBRARIES} ${PYTHON_LIBRARIES} Python3::NumPy carma)

# add_subdirectory(/usr/local/carma)
# set(CARMA_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/extern/carma/include")

# find_package(Armadillo REQUIRED)
# include(${PROJECT_SOURCE_DIR}/carma_config.cmake)
# message("Armadillo version found: ${ARMADILLO_VERSION_STRING}")
# 

# ## Make C++ extension, then python extension module
# add_library(_carma_svd MODULE src/carma_svd.cpp)
# python_extension_module(_carma_svd)


# ## Install extension modules
# # remove_flag_from_target(_carma_svd -Xarch_x86_64)
# include_directories(${PYTHON_INCLUDE_DIRS} ${CARMA_INCLUDE_DIRS} ${SOURCE_DIR})
# target_include_directories(carma_svd PUBLIC ${PYTHON_INCLUDE_DIRS} ${CARMA_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(carma_svd PUBLIC ${ARMADILLO_LIBRARIES} ${PYTHON_LIBRARIES} Python3::NumPy carma)
# target_compile_definitions(_carma_svd PUBLIC carma)














## ------- FAST_SVD TARGET -------
pybind11_add_module(fast_svd MODULE src/tallem/fast_svd.cpp)
#target_compile_options(fast_svd PUBLIC -O3 -Wall -std=c++17 -fPIC)

# -Wl,-undefined,dynamic_lookup
#target_link_options(fast_svd PUBLIC -Wl,-undefined,dynamic_lookup)

#target_include_directories(fast_svd PUBLIC ${PYTHON_INCLUDE_DIRS})
# target_link_libraries(fast_svd PUBLIC ${PYTHON_LIBRARIES} Python3::NumPy pybind11::headers ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

#pybind11_add_module(carma_svd MODULE src/tallem/carma_svd.cpp)
# message("LAPACK includes: ${LAPACK_LIBRARIES}")
# message("LAPACK linker flags: ${LAPACK_LINKER_FLAGS}")

## Install extension modules
# remove_flag_from_target(_carma_svd -Xarch_x86_64)
# include_directories(${PYTHON_INCLUDE_DIRS} ${CARMA_INCLUDE_DIRS} ${SOURCE_DIR})
# target_include_directories(carma_svd PUBLIC ${PYTHON_INCLUDE_DIRS} ${CARMA_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(carma_svd PUBLIC ${ARMADILLO_LIBRARIES} ${PYTHON_LIBRARIES} Python3::NumPy carma)


# target_compile_definitions(_carma_svd PUBLIC carma)

# target_compile_features(_carma_svd PUBLIC cxx_std_17)
# set_property(TARGET 01-1-hello-world PROPERTY CXX_STANDARD 17)

# install(TARGETS _example LIBRARY DESTINATION .)
# install(TARGETS fast_svd carma_svd LIBRARY DESTINATION .)
# install(TARGETS _carma_svd LIBRARY DESTINATION tallem_ext)