cpp = g++

cflags = -O3 -fPIC -DARMA_NO_DEBUG -DNDEBUG -Wno-deprecated -mtune=native -march=native -std=c++11 
ompflags = -fopenmp

include_common = -I./src 
include_pybind11 = `python3 -m pybind11 --includes`
include_eigen3 = -I${CONDA_PREFIX}/include/eigen3
include_arma = -I${CONDA_PREFIX}/include

INCLUDE = $(include_common) $(include_arma) \
          $(include_eigen3) $(include_pybind11)
loader = $(cpp)

libarma = -L/${CONDA_PREFIX}/lib -larmadillo
libomp = -lgomp

libs = $(libarma) $(libomp)

VPATH = src/compute src/polymlp src/longrange src/python

source = polymlp_gtinv_data.o polymlp_read_gtinv.o polymlp_basis_function.o \
         polymlp_functions_interface.o polymlp_model_params.o \
         polymlp_features.o polymlp_potential.o \
         polymlp_mlipkk_spherical_harmonics.o \
         local.o model.o py_model.o py_additive_model.o neighbor.o \
         pybind11_mlp.o ewald.o compute_features.o

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

