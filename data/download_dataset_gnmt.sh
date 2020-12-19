#!/bin/bash

mkdir gnmt
curl https://zenodo.org/record/3437893/files/gnmt_inference_data.zip --output gnmt/inference_data.zip
unzip gnmt/inference_data.zip
rm gnmt/inference_data.zip
