#!/usr/bin/env bash

set -exu

export PYTHONPATH=`pwd`

# first command-line argument: run-id
RUN_ID=$1
echo 'RUN_ID' = $RUN_ID

# second command-line aergument model-name
MODEL_NAME=$2
echo 'Running Model = ' $MODEL_NAME
if [ $MODEL_NAME != "Model1" -a $MODEL_NAME != "Model2" -a $MODEL_NAME != "Model3" ]
	then echo 'Error! Invalid Model'
	exit -1
fi

# third argument is number of arguments
NUM_EPOCHS=$3
echo 'Number of Epochs' = $NUM_EPOCHS
re_num='^[0-9]+$'
if ! [[ $NUM_EPOCHS =~ $re_num ]] ; then
   echo "Error! Not a number" >&2; exit 1
fi

# fourth argument is number of users to run
NUM_USERS=$4
echo 'Number of Users = ' $NUM_USERS
re_num='^[0-9]+$'
if ! [[ $NUM_USERS =~ $re_num ]] ; then
   echo "Error! Not a number" >&2; exit 1
fi

NUM_THREADS=14

export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

    mkdir -p ../logs/usc-isi/${TIME}

    sbatch -J TRAIN-MODEL-$MODEL_NUMBER-$MODEL_DESC \
                -e ../logs/usc-isi/${TIME}/train.err \
                -o ../logs/usc-isi/${TIME}/train.log \
                --cpus-per-task $NUM_THREADS \
                train-model.sh $RUN_ID $MODEL_NAME $NUM_EPOCHS $NUM_USERS
exit

