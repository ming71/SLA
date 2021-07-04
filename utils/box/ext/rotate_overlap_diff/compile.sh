#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}


cd cuda_op
if [ -d "build" ]; then
    rm -r build
fi
python setup.py install
cd ..
