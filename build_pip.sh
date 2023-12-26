#!/bin/zsh

version=3.10.8-intel
rm src/pypolymlp/cxx/lib
ln -s lib-$version src/pypolymlp/cxx/lib
python setup.py bdist_wheel
mv dist/*whl dist/python-$version/

version=3.11.3-intel
rm src/pypolymlp/cxx/lib
ln -s lib-$version src/pypolymlp/cxx/lib
python setup.py bdist_wheel
mv dist/*whl dist/python-$version/


rm src/pypolymlp/cxx/lib
ln -s lib-3.11.3-intel src/pypolymlp/cxx/lib


