#!/usr/bin/env bash

set -exu

export PYTHONPATH=`pwd`

# first command-line argument: run-id
RUN_ID=$1
echo 'RUN_ID = ' $RUN_ID

NUM_DIM=$2
echo 'Vector dimension = ' $NUM_DIM

NUM_EPOCHS=$3
echo 'Number of Epochs = ' $NUM_EPOCHS

NUM_THREADS=1

export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

    mkdir -p ../logs/usc-isi/${TIME}

    sbatch -J TRAIN-$NUM_DIM-E$NUM_EPOCHS \
                -e ../logs/${TIME}/train.err \
                -o ../logs/${TIME}/train.log \
                --cpus-per-task $NUM_THREADS \
                --mail-type=ALL \
                --mail-user=saran.krishna920@gmail.com \
                train-model.sh $RUN_ID $NUM_DIM $NUM_EPOCHS

exit

