cpp = g++

cflags = -O3 -fPIC -DARMA_NO_DEBUG -DNDEBUG -Wno-deprecated -mtune=native -march=native -std=c++11 
ompflags = -fopenmp

include_common = -I./src 
include_pybind11 = `python3 -m pybind11 --includes`
include_eigen3 = -I${CONDA_PREFIX}/include/eigen3
include_arma = -I${CONDA_PREFIX}/include
include_gsl = -I${CONDA_PREFIX}/include
include_boost = -I/home/seko/usr/local/boost/include

INCLUDE = $(include_common) $(include_boost) $(include_arma) \
          $(include_gsl) $(include_eigen3) $(include_pybind11)

loader = $(cpp)

libarma = -L/${CONDA_PREFIX}/lib -larmadillo
libgsl = -L/${CONDA_PREFIX}/lib -lgsl -lgslcblas
libboost = -L/home/seko/usr/local/boost/lib
libomp = -lgomp

libs = $(libarma) $(libboost) $(libgsl) $(libomp)

VPATH = src/compute src/polymlp src/longrange src/projector src/python

source = polymlp_gtinv_data.o polymlp_read_gtinv.o polymlp_basis_function.o \
         polymlp_functions_interface.o polymlp_model_params.o \
         polymlp_features.o polymlp_potential.o \
         polymlp_mlipkk_spherical_harmonics.o \
         local.o model.o py_model.o neighbor.o pybind11_mlp.o \
         projector.o ewald.o compute_features.o

all: mkdir $(addprefix lib/,$(source))

	$(loader) $(ldflags) -shared $(addprefix lib/,$(source)) \
        -o lib/mlpcpp.so $(libs)

mkdir: 
	@if [ ! -d lib ]; then \
        echo "mkdir lib"; mkdir lib; \
    fi 

lib/%.o: %.cpp
	$(cpp) -c $(cflags) $(ompflags) $(INCLUDE) -o $@ $<

clean:
	rm -rf lib

