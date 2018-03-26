#!/usr/bin/env bash
#
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --time=0-08:00         # Runtime in D-HH:MM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=62000

set -exu

RUN_ID=$1
MODEL_NAME=$2
NUM_EPOCHS=$3
NUM_USERS=$4

source ~/mypython/bin/activate
python main.py $RUN_ID $MODEL_NAME $NUM_EPOCHS $NUM_USERS


