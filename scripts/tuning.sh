#!/bin/bash

#BATCH --mail-type=ALL
#SBATCH --mail-user=elianna_kondylis@brown.edu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20G
#SBATCH -t 10:00:00

module load python/3.7.4
module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate /gpfs/data/rsingh47/chi3l1_venv
python ~/finalized_tuning.py
