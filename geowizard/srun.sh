#!/usr/bin/env bash

set -x

PARTITION=llmeval2
JOB_NAME=Hello
GPUS=$1 # ${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
# WORKSPACE=$2

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --quotatype=auto \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --mem-per-cpu=100000 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python run_infer_object.py \
    --input_dir input/wild_chouxiang \
    --output_dir output_chouxiang_ \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --domain 'object'
