#!/usr/bin/env bash

set -exu

export PYTHONPATH=`pwd`

# reads from the config file
CONFIG_FILE=$1
configs=$''
while IFS='' read -r line || [[ -n "$line" ]]; do
    IFS=':'
    read -ra key_val <<< "$line"
    configs=$configs"${key_val[1]}"$' '
done < $CONFIG_FILE

# reads the space separated values
IFS=' '
read -ra vals <<< "$configs"

# first command-line argument: run-id
RUN_ID="${vals[0]}"

# second command-line argument model-name
MODEL_NAME="${vals[1]}"
if [ $MODEL_NAME != "Model1" -a $MODEL_NAME != "Model2" -a $MODEL_NAME != "Model3" -a $MODEL_NAME != "Model4" -a $MODEL_NAME != "Model2Deeper" -a $MODEL_NAME != "Model3Deeper" ]
	then echo 'Error! Invalid Model'
	exit -1
fi

# third argument is number of arguments
NUM_EPOCHS="${vals[2]}"
re_num='^[0-9]+$'
if ! [[ $NUM_EPOCHS =~ $re_num ]] ; then
   echo "Error! Not a number" >&2; exit 1
fi

# fourth argument is number of users to run
NUM_USERS="${vals[3]}"
re_num='^[0-9]+$'
if ! [[ $NUM_USERS =~ $re_num ]] ; then
   echo "Error! Not a number" >&2; exit 1
fi

# 5th argument is number of hidden dimensions
HIDDEN_DIMS="${vals[4]}"

# 6th argument is the word corpus size
WORD_CORPUS_SIZE="${vals[5]}"

# 7th argument is the user dimensions
USR_DIMS="${vals[6]}"

# 8th argument is the email dimensions
EMAIL_DIMS="${vals[7]}"

NUM_THREADS=1

export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

    mkdir -p ../logs/usc-isi/${TIME}

    sbatch -J TRAIN-$MODEL_NAME-E$NUM_EPOCHS-U$NUM_USERS \
                -e ../logs/usc-isi/${TIME}/train.err \
                -o ../logs/usc-isi/${TIME}/train.log \
                --cpus-per-task $NUM_THREADS \
                --mail-type=ALL \
                --mail-user=mudit.bhargava90@gmail.com,mohit.surana95@gmail.com \
                train-model.sh $RUN_ID $MODEL_NAME $NUM_EPOCHS $NUM_USERS $HIDDEN_DIMS $WORD_CORPUS_SIZE $USR_DIMS $EMAIL_DIMS $TIME

exit

