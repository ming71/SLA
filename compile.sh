#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

clear

cd utils/box/ext/rbbox_overlap_gpu
if [ -d "build" ]; then
    rm -r build
fi
rm -r *.so 
python setup.py build_ext --inplace

cd ../rbbox_overlap_cpu
if [ -d "build" ]; then
    rm -r build
fi
rm  -rf *.so
python setup.py build_ext --inplace

cd ../rotate_overlap_diff
sh compile.sh



cd ../../../..
