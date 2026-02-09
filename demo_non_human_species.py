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
import connectome_models

def generate_slurm_script(num_tasks, script_path, formulation, species, dense_or_sparse):
    if species == "Marmoset":
        time_limit = 4
        memory_request = f"""#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G"""
    else:
        time_limit = 1
        memory_request = f""" """

    """ Generate the SLURM script for the job array. """
    slurm_script = f"""#!/bin/bash

#SBATCH --account=kg98
#SBATCH --output={cwd}/slurm_output/run-array_non-human_species_%A_%a.out

#SBATCH --array=0-{num_tasks-1}

{memory_request}

#SBATCH --time=0{time_limit}:00:00

echo "Processing Id" $SLURM_ARRAY_TASK_ID

echo "Activating virtual environment"

source {env_path}/bin/activate
conda activate {conda_env_name}

# Execute the Python script with the array index and arguments
python {script_path} --r_s_id $SLURM_ARRAY_TASK_ID --formulation {formulation} --path_data {path_data} --species {species} --dense_or_sparse {dense_or_sparse}
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

def generate_slurm_script_chunks(start_idx, end_idx, script_path, formulation, species, dense_or_sparse):
    
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

python {script_path} --r_s_id $SLURM_ARRAY_TASK_ID --formulation {formulation} --path_data {path_data} --species {species} --dense_or_sparse {dense_or_sparse}
echo "Done"
"""
    return "\n".join(headers) + body

def submit_slurm_jobs_chunks(num_tasks, script_path, formulation, path_data, species, dense_or_sparse, chunk_size=500):
    import math
    num_chunks = math.ceil(num_tasks / chunk_size)

    for i in range(num_chunks):
        print(i, "i chunk")
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size - 1, num_tasks - 1)

        slurm_script = generate_slurm_script_chunks(start_idx, end_idx, script_path, formulation, species, dense_or_sparse)
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
        _, _, mean_or_sum, _, _, _, _ = get_animal_paramameters(species)
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
            fixed_threshold_vertex = 0.6
            target_density = 0.9
        else:
            density_allen = "thr"
            fixed_threshold_vertex = 0.2
            target_density = "thr"

    if animal != "Mouse":
        density_allen = None

    resampling_weights = False

    return surface_name, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights

def get_non_human_species_EDR_parameters():

    eta_prob_connection_array = np.linspace(0, 15, 10000)
    eta_prob_connection_array_split = np.array_split(eta_prob_connection_array, 100)

    return eta_prob_connection_array, eta_prob_connection_array_split

def get_non_human_species_distance_atlas_parameters(species):

    if species == "Marmoset":
        eta = np.linspace(-15, 2, 10000)

    elif species == "Macaque":
        eta = np.linspace(-10, 2, 10000)

    elif species == "Mouse":
        eta = np.linspace(-15, 2, 10000)
    
    return eta

def get_non_human_species_matching_index_parameters(species):
    
    if species == "Marmoset":
        eta_list = np.linspace(-10, -1.2, 100)
        gamma_list = np.linspace(-0.5, 0.4, 100)

    elif species == "Mouse":
        eta_list = np.linspace(-9, -2, 100)
        gamma_list =  np.linspace(-0.05, 0.8, 100)
    
    elif species == "Macaque":
        eta_list = np.linspace(-5, -0.4, 100)
        gamma_list = np.linspace(-0.4, 0.5, 100)

    return eta_list, gamma_list

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

    if target_density == "thr" and species == "Mouse":
        pass
    
    elif target_density != 1:
        n_edges_empirical = int(target_density * len(idxes_parcel[0]))
        empirical_connectome = utilities.apply_threshold_to_match_densities(empirical_connectome, n_edges_empirical, idxes_parcel)

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

    # formulation = "GEM"
    formulation = "EDR-vertex"
    # formulation = "distance-atlas" 
    # formulation = "MI"

    _, _, _, r_s_values_list, _, _, _ = get_animal_paramameters(species, dense_or_sparse=dense_or_sparse) 
    
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
                species=species,
                dense_or_sparse=dense_or_sparse,
                chunk_size=chunk_size,  # adjust chunk size if needed
            )
            print("submitting chunks of a job array")

        else:
            # Generate and submit the SLURM job array
            slurm_script = generate_slurm_script(num_jobs, script_path, formulation, species, dense_or_sparse)
            submit_slurm_job(slurm_script)
            print("submitted job array")
    
    else:
        from non_human_species_models import  generate_and_save_model_performance

        for job_id in range(num_jobs):
            print("---------------------------------")
            print(job_id, "r_s_id")
            print("---------------------------------")
            generate_and_save_model_performance(species, dense_or_sparse, path_data, r_s_id=job_id, formulation=formulation)


def optimization_metric_species(species, dense_or_sparse):
    if species == "Mouse" and dense_or_sparse == "dense":
        return ["ranked_weights_strength", "spearman_union_weights"]
    if species == "Macaque" and dense_or_sparse == "dense":
        return ["ranked_weights_strength", "spearman_union_weights"]
    else:
        return ["degreeBinary", "ranked_weights_strength", "spearman_union_weights"]


def visualize_GEM_non_human_species_results(species, dense_or_sparse):
    """
    Visualize scatter plots for the GEM performance.
    """
    plot_connectivity_matrices = True

    lump = False

    cmap = utilities.get_colormap()

    color_optimized = cmap(0.8)
    color_not_optimized = color_optimized

    lump = False ## Fixed. Will override previous files if changed.

    species_parameters = get_animal_paramameters(species, dense_or_sparse)
    surface_name, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights = species_parameters
    
    loaded_parameters_and_variables = np.load(path_data + f"/{species}/{species}_parc_scheme={mean_or_sum}_saved_parameters_and_variables.npy", allow_pickle=True).item()
    mesh, _ = utilities.get_non_human_species_mesh(path_data, species)
    
    formulation = "GEM"

    directory = f"/{cwd}/data/results/non_human_species/{species}/resampled_weights_{resampling_weights}_formulation_{formulation}"
    k_range = np.array([k_ for k_ in range(2, 200)])

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "node connection distance", "clustering"]
    
    scatter_measures = ["degreeBinary", "common_weights", "ranked_weights_strength", "node connection distance", "clustering"]

    optimization_metric_list = optimization_metric_species(species, dense_or_sparse)

    measure_colors = {
    measure: (color_optimized if measure in optimization_metric_list or measure=="common_weights" else color_not_optimized)
    for measure in scatter_measures
    }

    dimension_files = (len(r_s_values_list), len(k_range))

    connectome_type = "atlas_species"
    fwhm = None
    heatmaps_dict, args_optimal = utilities.grab_results_heatmaps(optimization_metric_list, directory, network_measures, dimension_files, r_s_values_list, formulation, target_density, connectome_type, fwhm, plot_all=False, plot_opt=False)
    # plt.show() # set the arguments above to 'True' to visualize optimization landscape

    print()
    print("optimized network measures")
    for net_measure in network_measures:
        print(net_measure, f"opt score={heatmaps_dict[net_measure][args_optimal]}")
    print()

    best_r_s = r_s_values_list[args_optimal[0][0]]
    best_k = k_range[args_optimal[1][0]]

    print()
    print("optimized r_s = ", best_r_s)
    print("optimized k = ", best_k)
    print()

    if species != "Mouse":
        n_vertices = mesh.v.shape[0]
        idxes_cortex = loaded_parameters_and_variables['idxes_cortex']
        cortex_mask_array = np.zeros(n_vertices)
        cortex_mask_array[idxes_cortex] = 1
        cortex_mask_array = cortex_mask_array.astype(int)
        cortex_mask = True
    else:
        cortex_mask= False
        idxes_cortex = None

    model_parameters_and_variables = {}
    model_parameters_and_variables["target_density"] = target_density
    model_parameters_and_variables["species"] = species
    model_parameters_and_variables["density_allen"] = density_allen
    model_parameters_and_variables['fixed_threshold_vertex'] = fixed_threshold_vertex

    model_parameters_and_variables["characteristic_matrix"] = loaded_parameters_and_variables['characteristic_matrix']
    model_parameters_and_variables['vertices_in_connectome'] = loaded_parameters_and_variables["vertices_in_connectome"]
    model_parameters_and_variables['idxes_cortex'] = idxes_cortex
    model_parameters_and_variables['cortex_mask'] = cortex_mask

    print(model_parameters_and_variables['vertices_in_connectome'], "vertices in connectome")

    print(density_allen, "density allen")
    print(target_density, "target_density")
    
    _, empirical_parcel_connectivity, n_edges_empirical_parcel = get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables)
    
    model_parameters_and_variables['n_edges_empirical_parcel'] = n_edges_empirical_parcel

    evals, emodes = load_non_human_species_modes(species, lump=False)

    emodes = emodes[:, 0:k_range.max()+1]
    evals = evals[0:k_range.max()+1]

    model_parameters_and_variables["evals"] = evals
    model_parameters_and_variables["emodes"] = emodes

    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    print(fixed_threshold_vertex, "fixed_threshold_vertex")
    print(density, "density empirical")
    print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    print(cortex_mask, "cortex mask")

    distances, centroids = utilities.get_non_human_species_centroids(species, path_data)

    if plot_connectivity_matrices == True:
        utilities.plotConnectivity(empirical_parcel_connectivity, idxes_parcel, figsize=(13,11), original_cmap=cmap, show_cbar=True)
        plt.title("Empirical parcel connectome")
        # plt.show()

    # best_r_s = 0.49
    # best_k = 9
    geometric_model = connectome_models.generate_non_human_species_GEM(best_r_s, best_k, model_parameters_and_variables, resampling_weights=resampling_weights)
    
    if plot_connectivity_matrices == True:
        utilities.plotConnectivity(geometric_model, idxes_parcel, figsize=(13,11), original_cmap=cmap, show_cbar=True)
        plt.title("GEM")
        # plt.show()

    geometric_model_idxes = geometric_model[idxes_parcel]

    print(np.count_nonzero(geometric_model_idxes)/len(idxes_parcel[0]), "density model")
    # sys.exit()
    node_properties_model_dict = compute_node_properties(scatter_measures, geometric_model, distances)

    scatter_node_size = 70
    scatter_edge_size = 35

    alpha_node = 0.8
    alpha_edge = 0.6
    
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)

    utilities.plot_scatter_results(scatter_measures, geometric_model, geometric_model_idxes, empirical_parcel_connectivity_idxes, node_properties_model_dict, empirical_node_properties_dict, distances, scatter_node_size=scatter_node_size, scatter_edge_size=scatter_edge_size, alpha_node=alpha_node, alpha_edge=alpha_edge, measure_colors=measure_colors)
    plt.show()

def generate_non_human_species_comparison_results(species, which_results, dense_or_sparse):
    """
    Generate and save non-human species atlas-level model results.

    Models are generated one at a time.
    Stochastic models (EDR-vertex, Distance-atlas and Matching-index (MI)) are generated 100 times each.
    """

    list_of_number_of_communities = [3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    from utilities import get_performance_results

    formulation_GEM = "GEM"
    formulation_EDR_vertex = "EDR-vertex"
    formulation_distance_atlas = "distance-atlas"
    formulation_MI = "MI"
    formulation_Random = "Random"
    
    # Input (generate optimized results for one model at a time)
    ############################################################
    formulation_generate = formulation_GEM
    # formulation_generate = formulation_EDR_vertex
    # formulation_generate = formulation_distance_atlas
    # formulation_generate = formulation_MI
    # formulation_generate = formulation_Random
    ############################################################

    lump = False

    species_parameters = get_animal_paramameters(species, dense_or_sparse)
    surface_name, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights = species_parameters

    connectome_type = "atlas_species"
    fwhm = None
    
    loaded_parameters_and_variables = np.load(path_data + f"/{species}/{species}_parc_scheme={mean_or_sum}_saved_parameters_and_variables.npy", allow_pickle=True).item()
    mesh, _ = utilities.get_non_human_species_mesh(path_data, species)

    k_range = np.array([k_ for k_ in range(2, 200)])

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "node connection distance", "clustering"]
    network_measures_binary = [ "true_positive_rate", "degreeBinary", "node connection distance", "clustering"]
    

    ### Standard optimization metric used in the paper
    optimization_metric_list = ["degreeBinary", "ranked_weights_strength", "spearman_union_weights"]
    optimization_metric_list_binary = ["degreeBinary", "true_positive_rate"]
    
    opt_metric_str = "_".join(optimization_metric_list)
    opt_metric_str_binary = "_".join(optimization_metric_list_binary)

    # print(opt_metric_str, "network measures in the objective function")

    if species != "Mouse":
        n_vertices = mesh.v.shape[0]
        idxes_cortex = loaded_parameters_and_variables['idxes_cortex']
        cortex_mask_array = np.zeros(n_vertices)
        cortex_mask_array[idxes_cortex] = 1
        cortex_mask_array = cortex_mask_array.astype(int)
        cortex_mask = True
    else:
        cortex_mask= False
        idxes_cortex = None

    model_parameters_and_variables = {}
    model_parameters_and_variables["target_density"] = target_density
    model_parameters_and_variables["species"] = species
    model_parameters_and_variables["density_allen"] = density_allen
    model_parameters_and_variables['fixed_threshold_vertex'] = fixed_threshold_vertex

    model_parameters_and_variables["characteristic_matrix"] = loaded_parameters_and_variables['characteristic_matrix']
    model_parameters_and_variables['vertices_in_connectome'] = loaded_parameters_and_variables["vertices_in_connectome"]
    model_parameters_and_variables['idxes_cortex'] = idxes_cortex
    model_parameters_and_variables['cortex_mask'] = cortex_mask
    
    _, empirical_parcel_connectivity, n_edges_empirical_parcel = get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables)
    
    model_parameters_and_variables['n_edges_empirical_parcel'] = n_edges_empirical_parcel

    evals, emodes = load_non_human_species_modes(species, lump=False)

    emodes = emodes[:, 0:k_range.max()+1]
    evals = evals[0:k_range.max()+1]

    model_parameters_and_variables["evals"] = evals
    model_parameters_and_variables["emodes"] = emodes

    vertices = mesh.v

    vertices_in_connectome = loaded_parameters_and_variables["vertices_in_connectome"]

    if idxes_cortex is not None:
        vertices_species = vertices[idxes_cortex, :]

    elif vertices_in_connectome is not None:
        vertices_species = vertices[vertices_in_connectome, :]
    
    else:
        vertices_species = vertices

    distances_vertices = pdist(vertices_species)
    model_parameters_and_variables['distances_vertices'] = distances_vertices

    n_vertices = vertices_species.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)
    model_parameters_and_variables['idxes_vertex'] = idxes_vertex
    model_parameters_and_variables['n_vertices'] = n_vertices

    distances, centroids = utilities.get_non_human_species_centroids(species, path_data)
    distances /= np.max(distances)
    
    if which_results == "main":        
        dict_results = {net_measure:[] for net_measure in network_measures}
    elif which_results == "modularity":
        dict_results = {n_com:[] for n_com in list_of_number_of_communities}
    elif which_results == "spectral":
        list_results = []

    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)
    model_parameters_and_variables['idxes_parcel'] = idxes_parcel

    total_possible_connections = len(idxes_vertex[0])
    model_parameters_and_variables['total_possible_connections'] = total_possible_connections

    n_connections_vertex = int(fixed_threshold_vertex * total_possible_connections)
    model_parameters_and_variables['n_connections_vertex'] = n_connections_vertex

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)
    empirical_connectome_binary = (empirical_parcel_connectivity > 0).astype(float)
    n_edges_parcel_empirical = len(np.nonzero(empirical_parcel_connectivity_idxes)[0])

    print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    if which_results == "modularity":
        G_empirical = nx.from_numpy_array(empirical_connectome_binary)
        empirical_partition_dict = utilities.getDictOfPartitions(list_of_number_of_communities, G_empirical)
        empirical_labels_dict = utilities.labelsDict(G_empirical, empirical_partition_dict)
        empirical_partitions_set_dict = utilities.getDictOfPartitionsSet(empirical_partition_dict)

    elif which_results == "spectral":
        empirical_spectrum = utilities.compute_eigenspectrum(empirical_connectome_binary)

    directory = f"/{cwd}/data/results/non_human_species/{species}/resampled_weights_{resampling_weights}_formulation_{formulation_generate}"

    if formulation_generate == formulation_GEM:
        dimension_files_geo = (len(r_s_values_list), len(k_range))
        heatmaps_dict, args_optimal = utilities.grab_results_heatmaps(optimization_metric_list, directory, network_measures, dimension_files_geo, r_s_values_list, formulation_generate, target_density, connectome_type, fwhm)    
        
        best_r_s = r_s_values_list[args_optimal[0][0]]
        best_k = k_range[args_optimal[1][0]]
        best_params = (best_r_s, best_k)
        
        print(f"r_s={best_r_s}, k={best_k}")
                                               
        model = connectome_models.generate_non_human_species_GEM(best_r_s, best_k, model_parameters_and_variables, resampling_weights=resampling_weights)
        model_idxes = model[idxes_parcel]
        
        if which_results == "main":
            results = get_performance_results(network_measures, model, model_idxes, empirical_parcel_connectivity_idxes, empirical_node_properties_dict, distances)
            
            for net_measure in network_measures:
                dict_results[net_measure].append(results[net_measure])

        elif which_results == "modularity":
            model = (model > 0).astype(float)
            G_model = nx.from_numpy_array(model)            
            model_partition_dict = utilities.getDictOfPartitions(list_of_number_of_communities, G_model)
            model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
            nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)

            for n_com  in nvi_dict.keys():
                print(n_com, nvi_dict[n_com], "n_com NVI")
                dict_results[n_com].append(nvi_dict[n_com])

        elif which_results == "spectral":
            model = (model > 0).astype(int)
            model_spectrum = utilities.compute_eigenspectrum(model)
            spectral_distance = utilities.compute_spectral_distance(empirical_spectrum, model_spectrum)
            list_results.append(spectral_distance[0])
        
    elif formulation_generate == formulation_EDR_vertex:

        eta_prob_connection_array, _ = get_non_human_species_EDR_parameters()

        n_EDR_vertex_repetitions = 10
        number_of_repetitions = 100
        number_of_batches = 100
        moving_sum_average_heatmap = 0
        
        for repetition_id in range(n_EDR_vertex_repetitions):
            print(repet_id, "repet_id")            
            average_heatmap, heatmaps_dict = utilities.grab_non_human_species_EDR_results(formulation_generate, directory, network_measures, optimization_metric_list, target_density, number_of_batches, repetition_id)
            moving_sum_average_heatmap += average_heatmap
            
        moving_sum_average_heatmap /= n_EDR_vertex_repet
        args_optimal = np.where(moving_sum_average_heatmap == np.max(moving_sum_average_heatmap))
        best_eta_prob = eta_prob_connection_array[args_optimal[0][0]]

        for repet_ in range(number_of_repetitions):
            model = connectome_models.connectome_models.generate_EDR_non_human_species_model(best_eta_prob, model_parameters_and_variables)
            model_idxes = model[idxes_parcel]
            
            if which_results == "main":
                results = get_performance_results(network_measures, model, model_idxes, empirical_parcel_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.getDictOfPartitions(list_of_number_of_communities, G_model)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])

            elif which_results == "spectral":
                model = (model > 0).astype(int)
                model_spectrum = utilities.compute_eigenspectrum(model)
                spectral_distance = utilities.compute_spectral_distance(empirical_spectrum, model_spectrum)
                list_results.append(spectral_distance[0])

    elif formulation_generate == formulation_distance_atlas:

        number_of_repetitions = 100
        moving_sum_average_heatmap = 0
        count_ = 0
        eta = get_non_human_species_distance_atlas_parameters(species)

        number_of_distance_atlas_repet = 20
        
        for repetition_id in range(number_of_distance_atlas_repet):
            average_heatmap, _ = utilities.grab_distance_atlas_or_MI_heatmaps(optimization_metric_list_binary, directory, formulation_generate, target_density, connectome_type, fwhm, repetition_id, plot_heatmaps=False)
            moving_sum_average_heatmap += average_heatmap
            count_ +=1

        moving_sum_average_heatmap /= count_

        args_optimal = np.where(moving_sum_average_heatmap == np.max(moving_sum_average_heatmap))
        
        best_eta_distance_atlas = eta_distance_atlas[args_optimal[0][0]]

        cost_rule = powerlawRule
        
        for repet_ in range(number_of_repetitions):
            model = connectome_models.generate_distance_atlas_model(best_eta_distance_atlas, n_nodes, n_edges_parcel_empirical, distances, idxes_parcel, cost_rule)
            model_idxes = model[idxes_parcel]

            if which_results == "main":
                results = get_performance_results(network_measures_binary, model, model_idxes, empirical_parcel_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures_binary:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.getDictOfPartitions(list_of_number_of_communities, G_model)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])

            elif which_results == "spectral":
                model = (model > 0).astype(int)
                model_spectrum = utilities.compute_eigenspectrum(model)
                spectral_distance = utilities.compute_spectral_distance(empirical_spectrum, model_spectrum)
                list_results.append(spectral_distance[0])
    
    elif formulation_generate == formulation_MI:

        number_of_repetitions = 100
        moving_sum_average_heatmap = 0
        count_ = 0

        eta_MI, gamma_MI = get_non_human_species_matching_index_parameters(species)

        number_of_MI_repet = 20 
        
        for repetition_id in range(number_of_MI_repet):
            average_heatmap, _ = utilities.grab_distance_atlas_or_MI_heatmaps(optimization_metric_list_binary, directory, formulation_generate, target_density, connectome_type, fwhm, repetition_id, plot_heatmaps=False)
            moving_sum_average_heatmap += average_heatmap
            count_ +=1

        moving_sum_average_heatmap /= count_

        args_optimal = np.where(moving_sum_average_heatmap == np.max(moving_sum_average_heatmap))
        
        best_gamma_MI = gamma_MI[args_optimal[0][0]]
        best_eta_MI = eta_MI[args_optimal[1][0]]

        cost_rule = powerlawRule
        total_number_of_possible_edges = len(idxes_parcel[0])

        for repet_ in range(number_of_repetitions):
            model = connectome_models.generate_matching_index_model(best_eta_MI, best_gamma_MI, n_nodes, n_edges_parcel_empirical, distances, total_number_of_possible_edges, idxes_parcel, cost_rule)
            model_idxes = model[idxes_parcel]

            if which_results == "main":
                results = get_performance_results(network_measures_binary, model, model_idxes, empirical_parcel_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures_binary:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.getDictOfPartitions(list_of_number_of_communities, G_model)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])

            elif which_results == "spectral":
                model = (model > 0).astype(int)
                model_spectrum = utilities.compute_eigenspectrum(model)
                spectral_distance = utilities.compute_spectral_distance(empirical_spectrum, model_spectrum)
                list_results.append(spectral_distance[0])


    elif formulation_generate == formulation_Random:
        number_of_repetitions = 100
        for repet_ in range(number_of_repetitions):
            print(repet_, f"repet Random")
            model = connectome_models.generate_random_parcellated_model(n_vertices, total_possible_connections_vertex, n_connections_vertex, idxes_vertex, model_parameters_and_variables["characteristic_matrix"], idxes_parcel, n_edges_parcel_empirical, resampling_weights, weighted=False)
            model_idxes = model[idxes_parcel]
            
            if which_results == "main":
                results = get_performance_results(network_measures, model, model_idxes, empirical_parcel_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.getDictOfPartitions(list_of_number_of_communities, G_model)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])

            elif which_results == "spectral":
                model = (model > 0).astype(int)
                model_spectrum = utilities.compute_eigenspectrum(model)
                spectral_distance = utilities.compute_spectral_distance(empirical_spectrum, model_spectrum)
                list_results.append(spectral_distance[0])

    if formulation_generate == "MI" or formulation_generate == "distance-atlas":
        directory_save = directory + f"/optimized_for_{opt_metric_str_binary}"

    elif formulation_generate == "Random":
        directory_save = directory + f"/optimized_for_None"
    
    else:
        directory_save = directory + f"/optimized_for_{opt_metric_str}"

    directory_save += f"_target_density_{target_density}"

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    if which_results == "spectral":
        np.save(directory_save+ f"/{which_results}_optimized_results", list_results)
    else:
        np.save(directory_save+ f"/{which_results}_optimized_results", dict_results, allow_pickle=True)

    print("done and saved")


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
    # dense_or_sparse = "sparse"

    # 1. Generate the geometric eigenmodes
    # generate_geometric_modes(species)

    # 2. Optimize the GEM (explore parameters landscape)
    # optimize_and_save_non_human_species_results(species, dense_or_sparse)
    # print()

    #3. Visualize performance
    # visualize_GEM_non_human_species_results(species, dense_or_sparse)


    results = "main"
    # results = "modularity"
    # results = "spectral"
    
    #4. Generate benchmark models
    generate_non_human_species_comparison_results(species, which_results=results, dense_or_sparse=dense_or_sparse)

    #5. Compare GEM performance with other models
    # compare_non_human_species_models(species, which_results=results)


if __name__ == "__main__":
    mainFunction()