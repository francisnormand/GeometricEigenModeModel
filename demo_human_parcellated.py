import numpy as np
import lapy
from scipy.spatial.distance import pdist
import utilities
import sys
import os
import json
import subprocess
# from connectome_models import generate_high_res_GEM_humans, generate_high_res_LBO_humans, generate_EDR_vertex_model, generate_random_vertex_model
from network_measures_and_statistics import compute_node_properties
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def generate_slurm_script(num_tasks, script_path, formulation):
    """ Generate the SLURM script for the job array. """
    slurm_script = f"""#!/bin/bash

#SBATCH --account=kg98
#SBATCH --output={cwd}/slurm_output/run-array_human_parcellated_%A_%a.out

#SBATCH --array=0-{num_tasks-1}

#SBATCH --time=02:00:00

echo "Processing Id" $SLURM_ARRAY_TASK_ID

echo "Activating virtual environment"

source {env_path}/bin/activate
conda activate {conda_env_name}

# Execute the Python script with the array index and arguments
python {script_path} --r_s_id $SLURM_ARRAY_TASK_ID --formulation {formulation} --path_data {path_data}
"""
    return slurm_script

def submit_slurm_job(slurm_script):
    """ Submit the SLURM job. """
    slurm_file = "run_array.sh"
    with open(slurm_file, "w") as file:
        file.write(slurm_script)
    
    os.chmod(slurm_file, 0o755)
    subprocess.run(['sbatch', slurm_file], capture_output=True, text=True, check=True)

def wait_for_job(jobid):
    import time
    """Wait until jobid is no longer in the queue."""
    while True:
        result = subprocess.run(["squeue", "--job", jobid],
                                capture_output=True, text=True)
        if jobid not in result.stdout:
            break
        print(f"Waiting for job {jobid} to finish...")
        time.sleep(300)  # check every 5 minutes

def generate_slurm_script_chunks(start_idx, end_idx, script_path, formulation):
    
    headers = [
        "#!/bin/bash",
        "#SBATCH --account=kg98",
        f"#SBATCH --output={cwd}/slurm_output/run-array_human_parcellated_%A_%a.out",
        f"#SBATCH --array={start_idx}-{end_idx}",
        "#SBATCH --time=02:00:00"
    ]

    body = f"""
echo "Processing Id" $SLURM_ARRAY_TASK_ID

source {env_path}/bin/activate
conda activate {conda_env_name}

python {script_path} --r_s_id $SLURM_ARRAY_TASK_ID --formulation {formulation} --path_data {path_data}
"""
    return "\n".join(headers) + body

def submit_slurm_jobs_chunks(num_tasks, script_path, formulation, path_data, chunk_size=500):
    import math
    num_chunks = math.ceil(num_tasks / chunk_size)

    for i in range(num_chunks):
        print(i, "i chunk")
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size - 1, num_tasks - 1)

        slurm_script = generate_slurm_script_chunks(start_idx, end_idx, script_path, formulation)
        slurm_file = f"run_array_{i}.sh"
        with open(slurm_file, "w") as f:
            f.write(slurm_script)
        os.chmod(slurm_file, 0o755)

        print(f"Submitting job array {start_idx}-{end_idx}")
        result = subprocess.run(['sbatch', slurm_file], capture_output=True, text=True, check=True)
        stdout = result.stdout.strip()
        print(stdout)

        # get job ID from "Submitted batch job <ID>"
        jobid = stdout.split()[-1]
        wait_for_job(jobid)


def generate_geometric_modes(number_of_parcels=300):
    """
    Compute and save eigenvalues, eigenmodes, and (if masked) the B-matrix 
    of the human cortical surface.

    If the eigenmodes file already exists, the process is skipped.

    Parameters
    ----------
    path_data : str or Path
        Path to the data directory.
    lump : bool, default=False
        Whether to lump vertices during computation.
    cortex_mask : bool, default=True
        Whether the medial wall is masked.
    max_num_modes : int, default=500
        Maximum number of modes to compute.

    Outputs
    -------
    Saves the following files in `path_data`:
        - human_parcelalted_{number_of_parcels}_evals_lump_{lump}_masked_{cortex_mask}.npy
        - human_parcelalted_{number_of_parcels}_emodes_lump_{lump}_masked_{cortex_mask}.npy
        - human__parcelalted_{number_of_parcels}_Bmatrix_lump_{lump}_masked_{cortex_mask}.npy (if masked)
    """
    lump = False

    cortex_mask = True
    # cortex_mask = False
  
    max_num_modes = 500

    surface, surface_path = utilities.get_human_template_surface()

    output_eval_filename = path_data + f"/Schaefer{number_of_parcels}/human_parcellated_evals_lump_{lump}_masked_{cortex_mask}.npy"
    output_emode_filename = path_data + f"/Schaefer{number_of_parcels}/human_parcellated_emodes_lump_{lump}_masked_{cortex_mask}.npy"
    output_B_matrix_filename = None # Not saving the B matrix

    if os.path.exists(output_emode_filename):
        print(f"{output_emode_filename}")
        print()
        print("geometric modes already exists and will be overwritten")
        print()

    if cortex_mask is False: 
        evals, emodes = utilities.calc_surface_eigenmodes_nomask(surface_path, output_eval_filename, output_emode_filename, max_num_modes)
    else:
        cortex_mask_array = utilities.get_human_parcellated_cortex_mask(path_data, number_of_parcels)
        save_cut = 0  #Not saving temporary cut surface       
        evals, emodes, B_matrix = utilities.calc_surface_eigenmodes(surface_path, cortex_mask_array, output_eval_filename, output_emode_filename, output_B_matrix_filename, save_cut=save_cut, num_modes=max_num_modes, lump=lump)
    
    print()
    print("geometric modes were saved")

def get_human_parcellated_parameters(number_of_parcels):
    """
    Grab parameters associated with the empirical connectome and GEM.
    """

    connectome_type = "smoothed"
    # connectome_type = "unsmoothed"

    fwhm = 8
    if number_of_parcels == 300:
        target_density = 0.1

    fixed_vertex_threshold_density = 0.046
    
    resampling_weights=False

    cortex_mask = True
    # cortex_mask = False

    r_s_values_list = np.linspace(1, 20, 50)
    
    return r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, fixed_vertex_threshold_density, resampling_weights

def load_human_parcellated_modes(path_data, number_of_parcels, lump, cortex_mask):

    output_eval_filename = f"{path_data}/Schaefer{number_of_parcels}/human_parcellated_evals_lump_{lump}_masked_{cortex_mask}.npy"
    output_emode_filename = f"{path_data}/Schaefer{number_of_parcels}/human_parcellated_emodes_lump_{lump}_masked_{cortex_mask}.npy"

    return np.load(output_eval_filename), np.load(output_emode_filename)

def get_human_high_res_surface_and_parcellated_connectome(path_data, number_of_parcels, human_parcellated_parameters):
    
    surface, surface_path = utilities.get_human_template_surface()
    _, cortex_mask, connectome_type, fwhm, target_density, _, resampling_weights = human_parcellated_parameters

    empirical_connectome, _ = utilities.get_human_empirical_parcellated_connectome(path_data, target_density, number_of_parcels, connectome_type)
    
    if cortex_mask ==True:
        cortex_mask_array = utilities.get_human_parcellated_cortex_mask(path_data, number_of_parcels)
    else:
        cortex_mask_array = None

    return (surface, surface_path), cortex_mask_array, empirical_connectome

def optimize_and_save_human_parcellated_results(number_of_parcels=300):
    """
    Run large-scale optimization of the human parcellated models and save results.

    Depending on the formulation, this either:
    - Submits a SLURM job array (recommended, faster)
    - Or runs jobs sequentially in a loop (slower)

    Notes
    -----
    - Uses job arrays for EDR-vetex, distance-atlas, Matching index (MI) and GEM formulations.
    - Jobs call `human_vertex_models.generate_and_save_model_performance`.
    - For EDR, two chunks of jobs have to be sent, because there are 1000 jobs in total 
    """

    use_job_array = True 
    # use_job_array = False

    formulation = "GEM"
    # formulation = "EDR-vertex"
    # formulation = "distance-atlas" 
    # formulation = "MI"

    
    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, fixed_vertex_threshold_density, resampling_weights = get_human_parcellated_parameters(number_of_parcels)

    if formulation == "GEM":
        # GEM has 50 parameters, only once each
        num_jobs = len(r_s_values_list)
    elif formulation == "EDR-vertex":
        # EDR has 100 parameters, 10 times each
        num_jobs = 1000
        # separating into 2 chunks of 500 jobs
        chunk_size = 500
    else:
        # MI and distance-atlas
        num_jobs = 20
    
    if use_job_array == True:
        script_path = f"{cwd}/human_parcellated_models.py"
        print(script_path, "script_path")
        
        if num_jobs > 999:
            submit_slurm_jobs_chunks(
                num_tasks=num_jobs,
                script_path=script_path,
                formulation=formulation,
                path_data=path_data,
                chunk_size=chunk_size,  # adjust chunk size if needed
            )
            print("submitting chunks of a job array")

        else:
            # Generate and submit the SLURM job array
            slurm_script = generate_slurm_script(num_jobs, script_path, formulation)
            submit_slurm_job(slurm_script)
            print("submitted job array")
    
    else:
        from human_parcellated_models import generate_and_save_model_performance

        for job_id in range(num_jobs):
            print("---------------------------------")
            print(job_id, "r_s_id")
            print("---------------------------------")
            generate_and_save_model_performance(path_data, r_s_id=job_id, formulation=formulation)


# Current working director
cwd = os.getcwd()

# Global variable here. So it only is defined once. 
path_data = f"{cwd}/data/human_parcellated"

# Path to the Conda environment
conda_prefix = os.environ.get("CONDA_PREFIX")

env_path = conda_prefix.split("/conda/")[0]
# print(env_path, "base_conda_path")

# Current environment name
conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
# print(conda_env_name)


def mainFunction():
    """
    Main function to reproduce the analysis on the high-resolution human connectome.

    These functions have to be run sequentially.
    """
    number_of_parcels = 300

    # 1. Generate the geometric eigenmodes
    generate_geometric_modes(number_of_parcels)

    # 2. Optimize the GEM (explore parameters landscape)
    # optimize_and_save_human_parcellated_results(number_of_parcels)
    # print()

    #3. Visualize performance
    # visualize_GEM_human_parcellated_results()

    # results = "main"
    # results = "modularity"
    # results = "spectral"
    
    #4. Generate benchmark models
    # generate_human_parcellated_comparison_results(which_results=results)

    #5. Compare GEM performance with other models
    # compare_human_parcellated_models(which_results=results)


if __name__ == "__main__":
    mainFunction()