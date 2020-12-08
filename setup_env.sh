#!/bin/bash

pip install -r requirements.txt

git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference

cd mlperf_inference/loadgen

CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel

for f in dist/*; do
    pip install --force-reinstall $f
    break
done

echo "Setup done"
