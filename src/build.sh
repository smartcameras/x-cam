#!/bin/bash

################################################################################
echo "Configuring and building DBoW2 ..."

cd extern/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

################################################################################
echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

################################################################################
echo "Configuring and building VPR-P2P ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ..
