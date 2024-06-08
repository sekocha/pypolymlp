cmake_minimum_required(VERSION 3.13)

project(projlib_cmake CXX)

set(SRC_FILES
    src/projector/projector.cpp
    src/python/pybind11_mlp.cpp
)

#SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
#set(CMAKE_VERBOSE_MAKEFILE true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O2 -Wno-deprecated -DNDEBUG")
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)
message(STATUS ${EIGEN3_INCLUDE_DIR})
message(STATUS ${GSL_INCLUDE_DIR})

set(PYBIND11_PYTHON_VERSION
    3.9 3.10 3.11
)

#add_library(libprojcpp SHARED ${SRC_FILES})
pybind11_add_module(libprojcpp ${SRC_FILES})


target_include_directories(libprojcpp
  PUBLIC ${CMAKE_SOURCE_DIR}/src
  PUBLIC ${pybind11_INCLUDE_DIRS}
  PUBLIC ${EIGEN3_INCLUDE_DIR}
  PUBLIC ${GSL_INCLUDE_DIR}
)

target_link_libraries(libprojcpp
    PUBLIC ${OpenMP_CXX_LIBRARIES}
    GSL::gsl GSL::gslcblas
)


install( TARGETS libprojcpp
    COMPONENT python
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
