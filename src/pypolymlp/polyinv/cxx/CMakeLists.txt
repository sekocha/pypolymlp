cmake_minimum_required(VERSION 3.15)

project(projlib_cmake CXX)

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


message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
find_package(GSL REQUIRED)
message(STATUS "GSL include dir: ${GSL_INCLUDE_DIR}")
find_package(OpenMP)

set(SRC_FILES
    src/projector/projector.cpp
    src/python/pybind11_mlp.cpp
)


pybind11_add_module(libprojcpp ${SRC_FILES})

target_include_directories(
    libprojcpp
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(
      libprojcpp
      PRIVATE OpenMP::OpenMP_CXX
      PRIVATE Eigen3::Eigen
      PRIVATE pybind11::module
      PRIVATE GSL::gsl GSL::gslcblas
  )
else()
  target_link_libraries(
      libprojcpp
      PRIVATE Eigen3::Eigen
      PRIVATE pybind11::module
      PRIVATE GSL::gsl GSL::gslcblas
  )
  message(WARNING "OpenMP not found. It runs on single thread.")
endif()

install( TARGETS libprojcpp
    COMPONENT python
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
