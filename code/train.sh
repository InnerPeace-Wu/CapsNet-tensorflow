#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

TAG='log'

LOG="../logs/${TAG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time python ./train.py # --max_iters 10000 \
#    --data_dir 'path/to/data' \
#    --ckpt 'path/to/ckpt' \
#    --batch_size 100