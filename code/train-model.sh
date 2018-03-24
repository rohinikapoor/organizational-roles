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

MODEL_NUMBER=$1

source ~/mypython/bin/activate
python main.py $MODEL_NUMBER


