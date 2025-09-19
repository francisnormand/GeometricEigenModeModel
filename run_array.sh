#!/bin/bash

#SBATCH --account=kg98
#SBATCH --output=/fs04/kg98/FrancisN/scripts/GitHub/GeometricEigenModeModel/slurm_output/run-array_human_vertex_%A_%a.out

#SBATCH --array=0-49

#SBATCH --time=00:30:00
#SBATCH --qos=shortq

echo "Processing Id" $SLURM_ARRAY_TASK_ID

echo "Activating virtual environment"

source /scratch/kg98/FrancisN/miniconda/bin/activate
conda activate gt_new

# Execute the Python script with the array index and arguments
python /fs04/kg98/FrancisN/scripts/GitHub/GeometricEigenModeModel/human_vertex_models.py --r_s_id $SLURM_ARRAY_TASK_ID --formulation GEM --path_data /fs04/kg98/FrancisN/scripts/GitHub/GeometricEigenModeModel/data/human_high_res
