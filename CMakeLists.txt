cmake_minimum_required(VERSION 3.15...3.19)
project(tallem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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

# Armadillo
set(USE_ARMA_VERSION 10.5.2)
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ /usr/local/Cellar/ REQUIRED)

## Required CMake modules
find_package(Armadillo REQUIRED)
find_package(Python3 3.9 COMPONENTS Interpreter NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# add_subdirectory(src/carma)

## Make C++ extension, then python extension module
#add_library(_carma_svd MODULE src/carma_svd.cpp)
#python_extension_module(_carma_svd)
pybind11_add_module(_carma_svd MODULE src/carma_svd.cpp)
pybind11_add_module(_example MODULE src/example.cpp)


## Install extension modules
include_directories(${ARMADILLO_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_include_directories(_carma_svd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/carma)
target_link_libraries(_carma_svd PUBLIC ${ARMADILLO_LIBRARIES} ${PYTHON_LIBRARIES} Python3::NumPy)
target_compile_definitions(_carma_svd PUBLIC carma)
install(TARGETS _example _carma_svd LIBRARY DESTINATION .)
# install(TARGETS _carma_svd LIBRARY DESTINATION tallem_ext)

# pybind11_add_module(${MODNAME}
#     MODULE
#         arma_testing.cpp
# )
# TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC carma)

# TARGET_INCLUDE_DIRECTORIES(${MODNAME}
#     PUBLIC
#         $<INSTALL_INTERFACE:include>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# )

# TARGET_COMPILE_OPTIONS(${MODNAME}
#     PUBLIC
#         "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
# )

# TARGET_COMPILE_DEFINITIONS(${MODNAME}
#     PUBLIC
#         "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
# )

# # ##############################################################################
# #                                   INSTALL                                    #
# # ##############################################################################
# INSTALL(TARGETS ${MODNAME} DESTINATION examples)
# FILE(GLOB PY_EXAMPLE_FILES "${PROJECT_SOURCE_DIR}/examples/*.py")
# INSTALL(FILES ${PY_EXAMPLE_FILES} DESTINATION examples)
