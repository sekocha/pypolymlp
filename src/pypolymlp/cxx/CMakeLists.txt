cmake_minimum_required(VERSION 3.15)

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
    set(
        CMAKE_CXX_FLAGS_RELEASE
        "-O2 -Wno-deprecated -Wno-undefined-var-template -DNDEBUG -DEIGEN_NO_DEBUG"
    )
else()
    set(
        CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated -Wno-undefined-var-template -DEIGEN_NO_DEBUG"
    )
endif()
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

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
find_package(OpenMP)

set(SRC_FILES
    src/polymlp/polymlp_api.cpp
    src/polymlp/polymlp_basis_function.cpp
    src/polymlp/polymlp_features.cpp
    src/polymlp/polymlp_features_polynomial.cpp
    src/polymlp/polymlp_features_mapping_utils.cpp
    src/polymlp/polymlp_features_utils.cpp
    src/polymlp/polymlp_functions_interface.cpp
    src/polymlp/polymlp_mapping.cpp
    src/polymlp/polymlp_mlipkk_spherical_harmonics.cpp
    src/polymlp/polymlp_model_params.cpp
    src/polymlp/polymlp_model_params_gtinv.cpp
    src/polymlp/polymlp_model_params_polynomial.cpp
    src/polymlp/polymlp_potential.cpp
    src/polymlp/polymlp_products.cpp
    src/polymlp/polymlp_read_gtinv.cpp
    src/compute/neighbor.cpp
    src/compute/neighbor_half.cpp
    src/compute/neighbor_half_openmp.cpp
    src/compute/neighbor_cell.cpp
    src/compute/py_params.cpp
    src/compute/py_model.cpp
    src/compute/py_properties_fast.cpp
    src/compute/py_hybrid_model.cpp
    src/compute/py_features_attr.cpp
    src/compute/polymlp_eval.cpp
    src/compute/polymlp_eval_openmp.cpp
    src/compute/local.cpp
    src/compute/local_pair.cpp
    src/compute/model.cpp
    src/python/pybind11_mlp.cpp
    src/polymlp/polymlp_gtinv_data.cpp
    src/polymlp/polymlp_gtinv_data_order3.cpp
    src/polymlp/polymlp_gtinv_data_order4.cpp
    src/polymlp/polymlp_gtinv_data_order5.cpp
    src/polymlp/polymlp_gtinv_data_order6.cpp
    src/polymlp/polymlp_gtinv_data_v2_order1.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l0.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l1.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l2.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l3.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l4.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l5.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l6.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l7.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l8.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l9.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l10.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l11.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l12.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l13.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l14.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l15.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l16.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l17.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l18.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l19.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l20.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l21.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l22.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l23.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l24.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l25.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l26.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l27.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l28.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l29.cpp
    src/polymlp/polymlp_gtinv_data_v2_order2_l30.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l0.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l1.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l2.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l3.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l4.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l5.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l6.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l7.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l8.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l9.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l10.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l11.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l12.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l13.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l14.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l15.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l16.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l17.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l18.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l19.cpp
    src/polymlp/polymlp_gtinv_data_v2_order3_l20.cpp
)

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

install(
    TARGETS libmlpcpp
    COMPONENT python
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
