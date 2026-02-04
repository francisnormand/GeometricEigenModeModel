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
#SBATCH --output={cwd}/slurm_output/run-array_non-human_species_%A_%a.out

#SBATCH --array=0-{num_tasks-1}

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G 

#SBATCH --time=03:00:00

echo "Processing Id" $SLURM_ARRAY_TASK_ID

echo "Activating virtual environment"

source {env_path}/bin/activate
conda activate {conda_env_name}

# Execute the Python script with the array index and arguments
python {script_path} --r_s_id $SLURM_ARRAY_TASK_ID --formulation {formulation} --path_data {path_data}
echo "Done"
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
        f"#SBATCH --output={cwd}/slurm_output/run-array_non-human_species_%A_%a.out",
        f"#SBATCH --array={start_idx}-{end_idx}",
        "#SBATCH --time=02:00:00"
    ]

    body = f"""
echo "Processing Id" $SLURM_ARRAY_TASK_ID

source {env_path}/bin/activate
conda activate {conda_env_name}

python {script_path} --r_s_id $SLURM_ARRAY_TASK_ID --formulation {formulation} --path_data {path_data}
echo "Done"
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


def generate_geometric_modes(species):
    """
    Compute and save eigenvalues, eigenmodes of the 3 non-human species coritcal surfaces
    or volume (for the mouse isocortex).

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
        - {species}_evals_lump_{lump}.npy
        - {species}_emodes_lump_{lump}.npy
    """
    lump = False

    cortex_mask = True
    # cortex_mask = False
  
    max_num_modes = 500

    mesh, mesh_path = utilities.get_non_human_species_mesh(path_data, species)

    output_eval_filename = path_data + f"/{species}/{species}_evals_lump_{lump}.npy"
    output_emode_filename = path_data + f"/{species}/{species}_emodes_lump_{lump}.npy"
    output_B_matrix_filename = None # Not saving the B matrix

    if os.path.exists(output_emode_filename):
        print(f"{output_emode_filename}")
        print()
        print("geometric modes already exists and will be overwritten")
        print()

    if species == "Mouse":
        solver = lapy.Solver(mesh)
        eigenvalues, eigenmodes = solver.eigs(k=max_num_modes)

        np.save(output_eval_filename, eigenvalues)
        np.save(output_emode_filename, eigenmodes)
    
    else:
        _, _, _, _, mean_or_sum, _, _, _, _ = get_animal_paramameters(species)
        loaded_parameters_and_variables = np.load(path_data + f"/{species}/{species}_parc_scheme={mean_or_sum}_saved_parameters_and_variables.npy", allow_pickle=True).item()
        idxes_cortex = loaded_parameters_and_variables['idxes_cortex']
        n_vertices = mesh.v.shape[0]
        cortex_mask_array = np.zeros(n_vertices)
        cortex_mask_array[idxes_cortex] = 1
        cortex_mask_array = cortex_mask_array.astype(int)

        save_cut = 0  #Not saving temporary cut surface       
        evals, emodes, B_matrix = utilities.calc_surface_eigenmodes(mesh_path, cortex_mask_array, output_eval_filename, output_emode_filename, output_B_matrix_filename, save_cut=save_cut, num_modes=max_num_modes, lump=lump)
    
    print()
    print("geometric modes were saved")

def load_non_human_species_modes(species, lump=False):

    output_eval_filename = path_data + f"/{species}/{species}_evals_lump_{lump}.npy"
    output_emode_filename = path_data + f"/{species}/{species}_emodes_lump_{lump}.npy"

    return np.load(output_eval_filename), np.load(output_emode_filename)


def get_animal_paramameters(animal, dense_or_sparse="dense"):
    
    if animal == "ChimpYerkes29":
        surface_name="L.5k.ChimpYerkes29_midthickness-lh"
        mean_or_sum = "sum"
        r_s_values_list = np.linspace(0, 15, 50)
        target_density = 1
        fixed_threshold_vertex = 0.05 

    elif animal == "Marmoset":
        surface_name = "MBM_v3.0.1_midthickness-lh"
        mean_or_sum = "mean"
        r_s_values_list = np.linspace(0, 5, 50)
        target_density = 1
        fixed_threshold_vertex = 0.03 

    elif animal == "Macaque":
        surface_name = "MacaqueYerkes19_10k_midthickness-lh"
        mean_or_sum = "mean"
        r_s_values_list = np.linspace(0, 9, 50)
    
        if dense_or_sparse == "dense":
            target_density = 1
        else:
            target_density = 0.4
        
        fixed_threshold_vertex = 0.6 

    elif animal == "Mouse":
        surface_name = "rh_volume_ds_allen"
        mean_or_sum = "mean"
        r_s_values_list = np.linspace(0, 3, 50)
        
        if dense_or_sparse == "dense":
            density_allen = "raw"
            fixed_threshold_vertex = 0.2
            target_density = 1
        else:
            density_allen = "thr"
            fixed_threshold_vertex = 0.6
            target_density = 0.9

    if animal != "Mouse":
        density_allen = None

    resampling_weights = False

    return surface_name, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights

def get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables, representation="weighted", resampling_weights=None):

    target_density = model_parameters_and_variables['target_density']
    species = model_parameters_and_variables['species']
    density_allen = model_parameters_and_variables['density_allen']

    path_base_data = f"/home/fnormand/kg98_scratch/FrancisN/animal_data/{species}"

    extension_save = f"density={target_density}"

    if species == "Marmoset":
        empirical_connectome = np.loadtxt(path_base_data + "/" + "connectome_marmoset.txt", delimiter=",")

    elif species == "Macaque":
        empirical_connectome = np.loadtxt(path_base_data + "/" + "connectome_macaque.txt", delimiter=",")
    
    elif "Mouse" in species:
        path_connectome = f"rh_connectome_{density_allen}.txt"
        empirical_connectome = np.loadtxt(path_base_data + "/" + path_connectome, delimiter=",")

    n_nodes = empirical_connectome.shape[0]
    print(n_nodes, "n_nodes")
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_connectome = (empirical_connectome + empirical_connectome.T)/2
    np.fill_diagonal(empirical_connectome, 0)

    if target_density != 1:
        n_edges_empirical = int(target_density * len(idxes_parcel[0]))
        empirical_connectome = applyThresholdToMatchDensities(empirical_connectome, n_edges_empirical, idxes_parcel)

    empirical_connectome /= np.max(empirical_connectome)

    if representation == "binary":
        empirical_connectome[empirical_connectome != 0] = 1

    n_edges_empirical_parcel = len(np.nonzero(empirical_connectome[idxes_parcel])[0])

    if resampling_weights == "gaussian":
        empirical_connectome = resample_matrix(empirical_connectome)

    mesh, mesh_path = utilities.get_non_human_species_mesh(path_data, species)

    return mesh, empirical_connectome, n_edges_empirical_parcel

def optimize_and_save_non_human_species_results(species, dense_or_sparse):
    """
    Run large-scale optimization of the non-human species models and save results.

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

    surface_name, mesh_type, connectome_type, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights = get_animal_paramameters(species, dense_or_sparse=dense_or_sparse) 
    
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
        script_path = f"{cwd}/non_human_species_models.py"
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
        from non_human_species_models import  generate_and_save_model_performance

        for job_id in range(num_jobs):
            print("---------------------------------")
            print(job_id, "r_s_id")
            print("---------------------------------")
            generate_and_save_model_performance(species, dense_or_sparse, path_data, r_s_id=job_id, formulation=formulation)

# Current working director
cwd = os.getcwd()

# Global variable here. So it only is defined once. 
path_data = f"{cwd}/data/non_human_species"

# Path to the Conda environment
conda_prefix = os.environ.get("CONDA_PREFIX")

env_path = conda_prefix.split("/conda/")[0]
# print(env_path, "base_conda_path")

# Current environment name
conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
# print(conda_env_name)


def mainFunction():
    
    """
    Main function to reproduce the analysis on the non-human species connectomes.

    These functions have to be run sequentially.
    """

    species = "Mouse" # Has both dense and sparse
    # species = "Marmoset" # Dense only
    # species = "Macaque" # Has both dense and sparse

    dense_or_sparse = "dense"

    # 1. Generate the geometric eigenmodes
    # generate_geometric_modes(species)

    # 2. Optimize the GEM (explore parameters landscape)
    optimize_and_save_non_human_species_results(species, dense_or_sparse)
    # print()

    #3. Visualize performance
    # visualize_GEM_non_human_species_results(species)


    # results = "main"
    # results = "modularity"
    # results = "spectral"
    
    #4. Generate benchmark models
    # generate_non_human_species_comparison_results(species, which_results=results)

    #5. Compare GEM performance with other models
    # compare_non_human_species_models(species, which_results=results)


if __name__ == "__main__":
    mainFunction()