#!/bin/bash
#SBATCH --account=kg98
#SBATCH --output=/fs04/kg98/FrancisN/scripts/GitHub/GeometricEigenModeModel/slurm_output/run-array_human_vertex_%A_%a.out
#SBATCH --array=500-999
#SBATCH --time=00:30:00
#SBATCH --qos=shortq
echo "Processing Id" $SLURM_ARRAY_TASK_ID

source /fs04/kg98/FrancisN/scripts/python_env/miniconda/bin/activate
conda activate gt_new_new

python /home/fnormand/kg98/FrancisN/scripts/GitHub/GeometricEigenModeModel/human_vertex_models.py --r_s_id $SLURM_ARRAY_TASK_ID --formulation EDR --path_data /home/fnormand/kg98/FrancisN/scripts/GitHub/GeometricEigenModeModel/data/human_high_res
