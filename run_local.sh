#!/bin/bash

source ./run_common.sh
wandb login 4a30a34490a130dc21329fd04548bfd9f01cb1ec
common_opt="--config ../mlperf.conf"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=`pwd`/output/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
now=$(date +'%m%d%Y%H%M%S')
python main.py --profile $profile $common_opt \
    --model-name $model \
    --model $model_path $dataset \
    --output $OUTPUT_DIR $EXTRA_OPS $@ \
    --device $device 
