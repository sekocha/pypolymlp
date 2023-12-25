cmake_minimum_required(VERSION 3.13)

project(polymlplib_cmake CXX)

set ( SRC_FILES
    src/polymlp/polymlp_read_gtinv.cpp
    src/polymlp/polymlp_basis_function.cpp
    src/polymlp/polymlp_functions_interface.cpp
    src/polymlp/polymlp_model_params.cpp
    src/polymlp/polymlp_features.cpp
    src/polymlp/polymlp_potential.cpp
    src/polymlp/polymlp_mlipkk_spherical_harmonics.cpp
    src/compute/local.cpp
    src/compute/model.cpp
    src/compute/py_model.cpp
    src/compute/py_additive_model.cpp
    src/compute/neighbor.cpp
    src/compute/py_features_attr.cpp
    src/python/pybind11_mlp.cpp
    src/polymlp/polymlp_gtinv_data.cpp
)
message(STATUS ${CMAKE_SOURCE_DIR})

add_library(mlpcpp SHARED ${SRC_FILES})

target_compile_options(mlpcpp PUBLIC -O3 -Wno-deprecated -fopenmp)
target_compile_features(mlpcpp PUBLIC cxx_std_11)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED) # not found

target_include_directories(mlpcpp
  PUBLIC ${CMAKE_SOURCE_DIR}/src
  PUBLIC ${pybind11_INCLUDE_DIRS}
  PUBLIC ${Eigen3_INCLUDE_DIRS}
  PUBLIC $ENV{CONDA_PREFIX}/include/eigen3
  )

target_link_libraries(mlpcpp PUBLIC -lgomp)




