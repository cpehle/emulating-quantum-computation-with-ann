#!/bin/bash

wget https://github.com/nest/nest-simulator/archive/$1.tar.gz
tar xf $1.tar.gz
mkdir nest-simulator-$1-build
pushd .
cd nest-simulator-$1-build
cmake -DCMAKE_INSTALL_PREFIX:PATH=../nest-${1//v}/ ../nest-simulator-${1//v} # -Dwith-music=../music-${2//v}/ -Dwith-mpi=ON

make -j8
make install

popd

# remove intermediate files
rm $1.tar.gz
rm -rf nest-simulator-$1-build
rm -rf nest-simulator-${1//v}
