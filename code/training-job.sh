#!/usr/bin/env bash

set -exu

export PYTHONPATH=`pwd`

MODEL_NUMBER=$1
MODEL_DESC=$2

NUM_THREADS=14

export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

    mkdir -p logs/usc-isi/${TIME}

    sbatch -J TRAIN-MODEL-$MODEL_NUMBER-$MODEL_DESC \
                -e ../logs/usc-isi/${TIME}/train.err \
                -o ../logs/usc-isi/${TIME}/train.log \
                --cpus-per-task 1 \
                --ntasks-per-node $NUM_THREADS \
                train-model.sh $MODEL_NUMBER
exit

