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
NUM_DIM=$2
NUM_EPOCHS=$3

source ~/mypython/bin/activate
export PYTHONUNBUFFERED=TRUE
./demo.sh $NUM_DIM $NUM_EPOCHS

# mailx -S smtp=mail.cs.umass.edu -a ../logs/usc-isi/${TIME}/train.log -a ../logs/usc-isi/${TIME}/train.err -s "Test mail" msurana@cs.umass.edu < /dev/null
