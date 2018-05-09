#!/usr/bin/env bash
#
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --time=3-00:00         # Runtime in D-HH:MM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=62000

set -exu

RUN_ID=$1
MODEL_NAME=$2
NUM_EPOCHS=$3
NUM_USERS=$4
HIDDEN_DIMS=$5
WORD_CORPUS_SIZE=$6
USR_DIMS=$7
EMAIL_DIMS=$8
TIME=$9

source ~/mypython/bin/activate
export PYTHONUNBUFFERED=TRUE
python main.py $RUN_ID $MODEL_NAME $NUM_EPOCHS $NUM_USERS $HIDDEN_DIMS $WORD_CORPUS_SIZE $USR_DIMS $EMAIL_DIMS

# mailx -S smtp=mail.cs.umass.edu -a ../logs/usc-isi/${TIME}/train.log -a ../logs/usc-isi/${TIME}/train.err -s "Test mail" msurana@cs.umass.edu < /dev/null
