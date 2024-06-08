cpp = ${CONDA_PREFIX}/bin/g++

cflags = -O3 -fPIC -Wno-deprecated -std=c++11
ompflags = -fopenmp

include_common = -I./src
include_pybind11 = `python3 -m pybind11 --includes`
include_eigen3 = -I${CONDA_PREFIX}/include/eigen3
INCLUDE = $(include_common) $(include_eigen3) $(include_pybind11)

loader = $(cpp)

libomp = -lgomp
libs = $(libomp)

VPATH = src/compute src/polymlp src/python

source = polymlp_gtinv_data.o polymlp_read_gtinv.o polymlp_basis_function.o \
         polymlp_functions_interface.o polymlp_model_params.o \
         polymlp_features.o polymlp_potential.o \
         polymlp_mlipkk_spherical_harmonics.o \
         local.o model.o py_model.o py_additive_model.o neighbor.o \
         model_properties.o py_properties.o py_features_attr.o \
         neighbor_half.o neighbor_cell.o polymlp_eval.o \
         pybind11_mlp.o

all: mkdir $(addprefix lib/,$(source))

	$(loader) $(ldflags) -shared $(addprefix lib/,$(source)) \
        -o lib/libmlpcpp.so $(libs)

mkdir:
	@if [ ! -d lib ]; then \
        echo "mkdir lib"; mkdir lib; \
    fi

lib/%.o: %.cpp
	$(cpp) -c $(cflags) $(ompflags) $(INCLUDE) -o $@ $<

clean:
	rm -rf lib
