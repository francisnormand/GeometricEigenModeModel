#!/bin/bash

#SBATCH --account=kg98
#SBATCH --output=/fs04/kg98/FrancisN/scripts/GitHub/clean_GEM/slurm_output/run-array_human_vertex_%A_%a.out

#SBATCH --time=00:30:00
#SBATCH --qos=shortq

#SBATCH --array=0-49   # Range of tasks (adjust based on the number of subjects)

task_id=$((SLURM_ARRAY_TASK_ID))

echo "Processing Id :${task_id}"

echo "Activating virtual environment"

source /fs04/kg98/FrancisN/scripts/python_env/miniconda/bin/activate
conda activate gt_new_new

cd /home/fnormand/kg98/FrancisN/scripts/GitHub/clean_GEM

python human_vertex_models.py --r_s_id ${task_id}