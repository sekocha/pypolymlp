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
    src/compute/model_properties.cpp
    src/compute/py_model.cpp
    src/compute/py_properties.cpp
    src/compute/py_additive_model.cpp
    src/compute/neighbor.cpp
    src/compute/py_features_attr.cpp
    src/python/pybind11_mlp.cpp
    src/polymlp/polymlp_gtinv_data.cpp
)
message(STATUS ${CMAKE_SOURCE_DIR})

find_package(pybind11 REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS ${EIGEN3_INCLUDE_DIR})

set ( PYBIND11_PYTHON_VERSION 
    3.8 3.9 3.10 3.11
)
pybind11_add_module(libmlpcpp ${SRC_FILES})

target_compile_options(libmlpcpp PUBLIC -O3 -Wno-deprecated -fopenmp)
target_compile_features(libmlpcpp PUBLIC cxx_std_11)

target_include_directories(libmlpcpp
  PUBLIC ${CMAKE_SOURCE_DIR}/src
  PUBLIC ${pybind11_INCLUDE_DIRS}
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  )

target_link_libraries(libmlpcpp PUBLIC -lgomp)

install( TARGETS libmlpcpp 
    COMPONENT python
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)

