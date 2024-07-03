cmake_minimum_required(VERSION 3.13)

project(polymlplib_cmake CXX)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25")
    if(POLICY CMP0148)
        cmake_policy(SET CMP0148 NEW)
    endif()
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CXX_FLAGS_RELEASE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wno-deprecated -DNDEBUG")
endif()

if(NOT DEFINED CMAKE_CXX_FLAGS_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring Debug build")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring Release build")
else()
    message(STATUS "No specific build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE) set(CMAKE_VERBOSE_MAKEFILE true)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
# find_package(OpenMP REQUIRED)
find_package(OpenMP)

set(SRC_FILES
    src/compute/neighbor.cpp
    src/compute/neighbor_half.cpp
    src/compute/neighbor_cell.cpp
    src/compute/py_model.cpp
    src/compute/py_properties_fast.cpp
    src/compute/py_additive_model.cpp
    src/compute/py_features_attr.cpp
    src/compute/polymlp_eval.cpp
    src/compute/features.cpp
    src/compute/functions.cpp
    src/compute/local_fast.cpp
    src/compute/model_fast.cpp
    src/python/pybind11_mlp.cpp
    src/polymlp/polymlp_read_gtinv.cpp
    src/polymlp/polymlp_basis_function.cpp
    src/polymlp/polymlp_functions_interface.cpp
    src/polymlp/polymlp_model_params.cpp
    src/polymlp/polymlp_mlipkk_spherical_harmonics.cpp
    src/polymlp/polymlp_features.cpp
    src/polymlp/polymlp_potential.cpp
    src/polymlp/polymlp_gtinv_data.cpp
    src/polymlp/polymlp_gtinv_data_order1.cpp
    src/polymlp/polymlp_gtinv_data_order2.cpp
    src/polymlp/polymlp_gtinv_data_order3.cpp
    src/polymlp/polymlp_gtinv_data_order4.cpp
    src/polymlp/polymlp_gtinv_data_order5.cpp
    src/polymlp/polymlp_gtinv_data_order6.cpp
    src/polymlp/polymlp_gtinv_data_ver2.cpp
    src/polymlp/polymlp_gtinv_data_ver2_order1.cpp
    src/polymlp/polymlp_gtinv_data_ver2_order2.cpp
    src/polymlp/polymlp_gtinv_data_ver2_order3.cpp
    src/polymlp/polymlp_gtinv_data_ver2_order4.cpp
    src/polymlp/polymlp_gtinv_data_ver2_order5.cpp
    src/polymlp/polymlp_gtinv_data_ver2_order6.cpp
)

# add_library(libmlpcpp SHARED ${SRC_FILES})
pybind11_add_module(libmlpcpp ${SRC_FILES})

target_include_directories(libmlpcpp PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(OpenMP_CXX_FOUND)
  target_link_libraries(
      libmlpcpp
      PRIVATE OpenMP::OpenMP_CXX
      PRIVATE Eigen3::Eigen
      PRIVATE pybind11::module
  )
else()
  target_link_libraries(
      libmlpcpp
      PRIVATE Eigen3::Eigen
      PRIVATE pybind11::module
  )
  message(WARNING "OpenMP not found. It runs on single thread.")
endif()

#install(
#    TARGETS libmlpcpp
#    COMPONENT python
#    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
#)
install(
    TARGETS libmlpcpp
    COMPONENT python
    LIBRARY DESTINATION .
)
