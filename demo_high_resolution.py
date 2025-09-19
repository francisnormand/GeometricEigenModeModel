import numpy as np
import lapy
from scipy.spatial.distance import pdist
import utilities
import sys
import os
import json
import subprocess
from connectome_models import generate_high_res_GEM_humans, generate_high_res_LBO_humans, generate_EDR_vertex_model, generate_random_vertex_model
from network_measures_and_statistics import compute_node_properties
import matplotlib.pyplot as plt
import networkx as nx

def generate_slurm_script(num_tasks, script_path, formulation):
    """ Generate the SLURM script for the job array. """
    slurm_script = f"""#!/bin/bash

#SBATCH --account=kg98
#SBATCH --output={cwd}/slurm_output/run-array_human_vertex_%A_%a.out

#SBATCH --array=0-{num_tasks-1}

#SBATCH --time=00:30:00
#SBATCH --qos=shortq

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
        f"#SBATCH --output={cwd}/slurm_output/run-array_human_vertex_%A_%a.out",
        f"#SBATCH --array={start_idx}-{end_idx}",
        "#SBATCH --time=00:30:00",
        "#SBATCH --qos=shortq"
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


def get_human_high_res_surface_and_connectome(path_data, human_vertex_parameters):
    """
    Load the high-resolution human cortical surface and vertex-level empirical connectome.

    Parameters
    ----------
    path_data : str or Path
        Path to the data directory.
    human_vertex_parameters : tuple
        Parameters for loading the connectome, expected as:
        (_, cortex_mask, connectome_type, fwhm, target_density, resampling_weights)

    Returns
    -------
    surface_info : tuple
        (surface, surface_path) of the human cortical template.
    cortex_mask_array : ndarray or None
        Cortical mask if `cortex_mask` is True, else None.
    empirical_connectome : ndarray
        Vertex-level empirical connectome.
    """

    surface, surface_path = utilities.get_human_template_surface(path_data)
    _, cortex_mask, connectome_type, fwhm, target_density, resampling_weights = human_vertex_parameters

    empirical_connectome = utilities.get_human_empirical_vertex_connectome(path_data, connectome_type, fwhm, target_density, resampling_weights, npz_=False)
    
    if cortex_mask ==True:
        cortex_mask_array = utilities.get_human_cortex_mask(path_data)
    else:
        cortex_mask_array = None

    return (surface, surface_path), cortex_mask_array, empirical_connectome
    

def load_human_vertex_modes(path_data, lump, cortex_mask):
    """
    Load precomputed eigenvalues and eigenmodes of the human cortical surface.

    Parameters
    ----------
    path_data : str or Path
        Path to the data directory.
    lump : str or int
        Identifier for the lumping/resolution used.
    cortex_mask : bool
        Whether the modes were computed with cortical masking.

    Returns
    -------
    evals : ndarray
        Array of eigenvalues.
    emodes : ndarray
        Array of eigenmodes.
    """

    output_eval_filename = path_data + f"/human_high_res_evals_lump_{lump}_masked_{cortex_mask}.npy"
    output_emode_filename = path_data + f"/human_high_res_emodes_lump_{lump}_masked_{cortex_mask}.npy"

    return np.load(output_eval_filename), np.load(output_emode_filename)

def generate_geometric_modes():
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
        - human_high_res_evals_lump_{lump}_masked_{cortex_mask}.npy
        - human_high_res_emodes_lump_{lump}_masked_{cortex_mask}.npy
        - human_high_res_Bmatrix_lump_{lump}_masked_{cortex_mask}.npy (if masked)
    """
    lump = False

    cortex_mask = True
    # cortex_mask = False

    max_num_modes = 500

    surface, surface_path = utilities.get_human_template_surface(path_data)

    output_eval_filename = path_data + f"/human_high_res_evals_lump_{lump}_masked_{cortex_mask}.npy"
    output_emode_filename = path_data + f"/human_high_res_emodes_lump_{lump}_masked_{cortex_mask}.npy"
    # output_B_matrix_filename = path_data + f"/human_high_res_Bmatrix_lump_{lump}_masked_{cortex_mask}.npy"
    output_B_matrix_filename = None # Not saving the B matrix

    if os.path.exists(output_emode_filename):
        print(f"{output_emode_filename}")
        print()
        print("geometric modes already exists and will be overwritten")
        print()

    if cortex_mask is False: 
        evals, emodes = utilities.calc_surface_eigenmodes_nomask(surface_path, output_eval_filename, output_emode_filename, max_num_modes)
    else:
        cortex_mask_array = utilities.get_human_cortex_mask(path_data)
        save_cut = 0  #Not saving temporary cut surface       
        evals, emodes, B_matrix = utilities.calc_surface_eigenmodes(surface_path, cortex_mask_array, output_eval_filename, output_emode_filename, output_B_matrix_filename, save_cut=save_cut, num_modes=max_num_modes, lump=lump)
    
    print()
    print("geometric modes were saved")

def optimize_and_save_human_high_resolution_results():
    """
    Run large-scale optimization of the human high-resolution model and save results.

    Depending on the formulation, this either:
    - Submits a SLURM job array (recommended, faster)
    - Or runs jobs sequentially in a loop (slower)

    Notes
    -----
    - Uses job arrays for EDR and GEM formulations.
    - Runs sequentially for LBO (no r_s parameter).
    - Jobs call `human_vertex_models.generate_and_save_model_performance`.
    - For EDR, two chunks of jobs have to be sent, because there are 1000 jobs in total 
    """

    
    use_job_array = True 
    # use_job_array = False

    formulation = "GEM"
    # formulation = "LBO" 
    # formulation = "EDR"

    if formulation == "LBO":
        use_job_array = False # LBO has no r_s parameter, so no need for a job array.

    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, resampling_weights = get_human_vertex_parameters()

    if formulation == "GEM":
        # GEM has 50 parameters, only once each
        num_jobs = len(r_s_values_list)
    else:
        # EDR has 100 parameters, 10 times each
        num_jobs = 1000
        # separating into 2 chunks of 500 jobs
        chunk_size = 500
    
    if use_job_array == True:
        script_path = f"{cwd}/human_vertex_models.py"
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
        from human_vertex_models import generate_and_save_model_performance
        
        if formulation == "LBO":
               generate_and_save_model_performance(path_data, r_s_id=None, formulation=formulation)

        else:
            for job_id in range(num_jobs):
                print("---------------------------------")
                print(job_id, "r_s_id")
                print("---------------------------------")
                generate_and_save_model_performance(path_data, r_s_id=job_id, formulation=formulation)


def visualize_GEM_human_vertex_results(plot_connectivity_matrices=False):
    """
    Visualize scatter plots for the GEM performance.
    """

    lump = False

    cmap = utilities.get_colormap()

    color_optimized = cmap(0.8)
    color_not_optimized = color_optimized

    human_vertex_parameters = get_human_vertex_parameters()
    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, resampling_weights = human_vertex_parameters

    (surface, _), cortex_mask_array, empirical_vertex_connectivity = get_human_high_res_surface_and_connectome(path_data, human_vertex_parameters)

    formulation = "GEM"

    directory = f"{cwd}/data/results/human_high_resolution/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"

    k_range = np.array([k_ for k_ in range(2, 200)])

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "node connection distance", "clustering"]
    
    scatter_measures = ["degreeBinary", "common_weights", "ranked_weights_strength", "node connection distance"]

    optimization_metric_list = ["degreeBinary", "ranked_weights_strength", "spearman_union_weights"]

    measure_colors = {
    measure: (color_optimized if measure in optimization_metric_list or measure=="common_weights" else color_not_optimized)
    for measure in scatter_measures
    }

    dimension_files = (len(r_s_values_list), len(k_range))

    heatmaps_dict, args_optimal = utilities.grab_human_vertex_heatmaps(optimization_metric_list, directory, network_measures, dimension_files, r_s_values_list, formulation, target_density, connectome_type, fwhm, plot_all=False, plot_opt=False)
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

    if cortex_mask ==True:
        cortex_mask_array = utilities.get_human_cortex_mask(path_data)
        idxes_cortex = np.where(cortex_mask_array == 1)[0]
    
    else:
        cortex_mask_array = None
    
    evals_geo, emodes_geo  = load_human_vertex_modes(path_data, lump, cortex_mask)

    emodes_geo = emodes_geo[:, 0:k_range.max()+1]
    evals_geo = evals_geo[0:k_range.max()+1]

    vertices = surface.v

    if cortex_mask ==True:
        emodes_geo = emodes_geo[idxes_cortex, :]
        vertices = vertices[idxes_cortex, :]
    
    distances = pdist(vertices)

    n_vertices = emodes_geo.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)

    empirical_vertex_connectivity_idxes = empirical_vertex_connectivity[idxes_vertex]
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_vertex_connectivity, distances)

    empirical_connectome_binary = (empirical_vertex_connectivity > 0).astype(int)
    n_edges_vertex_empirical = len(np.nonzero(empirical_vertex_connectivity_idxes)[0])

    if plot_connectivity_matrices == True:
        utilities.plotConnectivity(empirical_vertex_connectivity, idxes_vertex, figsize=(13,11), original_cmap=cmap, show_cbar=True)
        plt.title("Empirical vertex connectome")
        # plt.show()
                                                                    
    geometric_model = generate_high_res_GEM_humans(best_r_s, best_k, emodes_geo, evals_geo, target_density, idxes_vertex, resampling_weights)
    
    if plot_connectivity_matrices == True:
        utilities.plotConnectivity(geometric_model, idxes_vertex, figsize=(13,11), original_cmap=cmap_, show_cbar=True)
        plt.title("GEM")
        plt.show()

    geometric_model_idxes = geometric_model[idxes_vertex]
    node_properties_model_dict = compute_node_properties(scatter_measures, geometric_model, distances)

    utilities.plot_human_vertex_scatter_splots(scatter_measures, geometric_model, geometric_model_idxes, empirical_vertex_connectivity_idxes, node_properties_model_dict, empirical_node_properties_dict, distances, measure_colors)
    plt.show()
    
def get_human_vertex_parameters():
    """
    Grab parameters associated with the empirical connectome and GEM.
    """

    connectome_type = "smoothed"
    # connectome_type = "unsmoothed"

    fwhm = 8
    target_density = 0.046
    resampling_weights=False

    cortex_mask = True
    # cortex_mask = False

    r_s_values_list = np.linspace(1, 20, 50)
    
    return r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, resampling_weights

def get_human_vertex_EDR_parameters():
    """
    Grab parameters associated with the EDR model.
    """
    eta_prob_connection_array = np.linspace(0.4, 0.7, 100)
    eta_weights_array = np.linspace(-0.1, 0.3, 100)

    return eta_prob_connection_array, eta_weights_array


def generate_human_vertex_comparison_results(which_results="main"):
    """
    Generate and save human vertex-level model results.

    Models are generated one at a time.
    Stochastic models (EDR and Permuted) are generated 100 times each.
    """

    list_of_number_of_communities = [3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    from human_vertex_models import get_human_vertex_results

    formulation_GEM = "GEM"
    formulation_LBO = "LBO"
    formulation_permuted_evals = "permuted_eigenvalues_order"
    formulation_EDR = "EDR"
    formulation_Random = "Random"
    
    # Input (generate optimized results for one model at a time)
    ############################################################
    # formulation_generate = formulation_GEM
    # formulation_generate = formulation_LBO
    # formulation_generate = formulation_permuted_evals
    # formulation_generate = formulation_EDR
    formulation_generate = formulation_Random
    ############################################################

    lump = False

    human_vertex_parameters = get_human_vertex_parameters()
    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, resampling_weights = human_vertex_parameters

    (surface, _), cortex_mask_array, empirical_vertex_connectivity = get_human_high_res_surface_and_connectome(path_data, human_vertex_parameters)

    target_density = 0.046

    k_range = np.array([k_ for k_ in range(2, 200)])

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "node connection distance", "clustering"]
    distance_measures = []

    ### Standard optimization metric used in the paper
    optimization_metric_list = ["degreeBinary", "ranked_weights_strength", "spearman_union_weights"]
    
    opt_metric_str = "_".join(optimization_metric_list)

    print(opt_metric_str, "network measures in the objective function")

    if cortex_mask ==True:
        cortex_mask_array = utilities.get_human_cortex_mask(path_data)
        idxes_cortex = np.where(cortex_mask_array == 1)[0]
    
    else:
        cortex_mask_array = None
    
    evals_geo, emodes_geo  = load_human_vertex_modes(path_data, lump, cortex_mask)

    emodes_geo = emodes_geo[:, 0:k_range.max()+1]
    evals_geo = evals_geo[0:k_range.max()+1]

    vertices = surface.v

    if cortex_mask ==True:
        emodes_geo = emodes_geo[idxes_cortex, :]
        vertices = vertices[idxes_cortex, :]

    
    n_vertices = emodes_geo.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)
    
    if which_results == "main":        
        dict_results = {net_measure:[] for net_measure in network_measures}
    elif which_results == "modularity":
        dict_results = {n_com:[] for n_com in list_of_number_of_communities}

    total_possible_connections = len(idxes_vertex[0])

    distances = pdist(vertices)

    empirical_vertex_connectivity_idxes = empirical_vertex_connectivity[idxes_vertex]
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_vertex_connectivity, distances)
    empirical_connectome_binary = (empirical_vertex_connectivity > 0).astype(int)
    n_edges_vertex_empirical = len(np.nonzero(empirical_vertex_connectivity_idxes)[0])

    if which_results == "modularity":
        G_empirical = nx.from_numpy_array(empirical_connectome_binary)
        empirical_partition_dict = utilities.efficient_newman_spectral_communities(G_empirical, list_of_number_of_communities)
        empirical_labels_dict = utilities.labelsDict(G_empirical, empirical_partition_dict)
        empirical_partitions_set_dict = utilities.getDictOfPartitionsSet(empirical_partition_dict)

    directory = f"{cwd}/data/results/human_high_resolution/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation_generate}"

    if formulation_generate == formulation_GEM:
        dimension_files_geo = (len(r_s_values_list), len(k_range))
        heatmaps_dict, args_optimal = utilities.grab_human_vertex_heatmaps(optimization_metric_list, directory, network_measures, dimension_files_geo, r_s_values_list, formulation_generate, target_density, connectome_type, fwhm)    
        
        best_r_s = r_s_values_list[args_optimal[0][0]]
        best_k = k_range[args_optimal[1][0]]
        best_params = (best_r_s, best_k)
        
        print(f"r_s={best_r_s}, k={best_k}")
        model = generate_high_res_GEM_humans(best_r_s, best_k, emodes_geo, evals_geo, target_density, idxes_vertex, resampling_weights)
        model_idxes = model[idxes_vertex]
        
        if which_results == "main":
            results = get_human_vertex_results(network_measures, model, model_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances)
            
            for net_measure in network_measures:
                dict_results[net_measure].append(results[net_measure])

        elif which_results == "modularity":
            model = (model > 0).astype(int)
            G_model = nx.from_numpy_array(model)
            model_partition_dict = utilities.efficient_newman_spectral_communities(G_model, list_of_number_of_communities)
            model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
            nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
            for n_com  in nvi_dict.keys():
                dict_results[n_com].append(nvi_dict[n_com])
        

    elif formulation_generate == formulation_LBO:

        heatmaps_dict, args_optimal = utilities.grab_human_vertex_LBO_heatmaps(optimization_metric_list, directory, network_measures, formulation_generate, target_density, connectome_type, fwhm)    
        best_k = k_range[args_optimal[0][0]]

        best_params = (best_k)
        print(f" LBO, best k={best_k}")

        model = generate_high_res_LBO_humans(None, best_k, emodes_geo, evals_geo, target_density, idxes_vertex, resampling_weights)
        model_idxes = model[idxes_vertex]
        
        if which_results == "main":
            results = get_human_vertex_results(network_measures, model, model_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances)
            
            for net_measure in network_measures:
                dict_results[net_measure].append(results[net_measure])

        elif which_results == "modularity":
            model = (model > 0).astype(int)
            G_model = nx.from_numpy_array(model)
            model_partition_dict = utilities.efficient_newman_spectral_communities(G_model, list_of_number_of_communities)
            model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
            nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
            for n_com  in nvi_dict.keys():
                dict_results[n_com].append(nvi_dict[n_com])


    elif formulation_generate == formulation_permuted_evals:

        dimension_files_geo = (len(r_s_values_list), len(k_range))
        # Loading the results from formulation_GEM.
        directory = f"{cwd}/data/results/human_high_resolution/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation_GEM}"
        heatmaps_dict, args_optimal = utilities.grab_human_vertex_heatmaps(optimization_metric_list, directory, network_measures, dimension_files_geo, r_s_values_list, formulation_GEM, target_density, connectome_type, fwhm)    
        
        best_r_s = r_s_values_list[args_optimal[0][0]]
        best_k = k_range[args_optimal[1][0]]
        best_params = 0

        print(best_k, "best k ")
        print(best_r_s, "best r_s ")

        evals_geo_k = evals_geo[0:best_k]
        number_of_repetitions = 100

        for repet_ in range(number_of_repetitions):
            print(repet_, "repet shuffled evals")
            evals_geo_k_copy = np.copy(evals_geo_k) 
            np.random.shuffle(evals_geo_k_copy)

            model = generate_high_res_GEM_humans(best_r_s, best_k, emodes_geo, evals_geo_k_copy, target_density, idxes_vertex, resampling_weights)
            model_idxes = model[idxes_vertex]
            
            if which_results == "main":
                results = get_human_vertex_results(network_measures, model, model_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.efficient_newman_spectral_communities(G_model, list_of_number_of_communities)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])


    elif formulation_generate == formulation_EDR:
        eta_prob_connection_array, eta_weights_array = get_human_vertex_EDR_parameters()
        dimension_files_EDR = (len(eta_prob_connection_array), len(eta_weights_array))

        n_EDR_vertex_repet = 10
        number_of_repetitions = 100
        moving_sum_average_heatmap = 0

        for repet_id in range(n_EDR_vertex_repet):
            print(repet_id, "repet_id")
            heatmaps_dict, average_heatmap = utilities.grab_human_EDR_heatmaps(repet_id, optimization_metric_list, directory, network_measures, dimension_files_EDR, eta_prob_connection_array, formulation_generate, target_density, connectome_type, fwhm, plot_heatmaps=True)    
            plt.show()
            moving_sum_average_heatmap += average_heatmap
            
        moving_sum_average_heatmap /= n_EDR_vertex_repet
        args_optimal = np.where(moving_sum_average_heatmap == np.max(moving_sum_average_heatmap))
        best_eta_prob, best_eta_weights = eta_prob_connection_array[args_optimal[0][0]], eta_weights_array[args_optimal[1][0]]

        best_params = (best_eta_prob, best_eta_weights)
        for repet_ in range(number_of_repetitions):
            print(repet_, f"repet EDR")
            model = generate_EDR_vertex_model(best_eta_prob, best_eta_weights, distances, idxes_vertex, n_vertices, n_edges_vertex_empirical, total_possible_connections, resampling_weights)
            model_idxes = model[idxes_vertex]
            
            if which_results == "main":
                results = get_human_vertex_results(network_measures, model, model_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.efficient_newman_spectral_communities(G_model, list_of_number_of_communities)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])

    elif formulation_generate == formulation_Random:
        number_of_repetitions = 100
        best_params = 0
        for repet_ in range(number_of_repetitions):
            print(repet_, f"repet Random")
            model = generate_random_vertex_model(n_vertices, total_possible_connections, n_edges_vertex_empirical, idxes_vertex, weighted=True)
            model_idxes = model[idxes_vertex]
            
            if which_results == "main":
                results = get_human_vertex_results(network_measures, model, model_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances)
            
                for net_measure in network_measures:
                    dict_results[net_measure].append(results[net_measure])

            elif which_results == "modularity":
                model = (model > 0).astype(int)
                G_model = nx.from_numpy_array(model)
                model_partition_dict = utilities.efficient_newman_spectral_communities(G_model, list_of_number_of_communities)
                model_labels_dict = utilities.labelsDict(G_model, model_partition_dict)
                nvi_dict = utilities.getDictOfNVI(empirical_labels_dict, model_labels_dict)
                for n_com  in nvi_dict.keys():
                    dict_results[n_com].append(nvi_dict[n_com])

    
    
    directory_save = directory + f"/{opt_metric_str}"
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    np.save(directory_save+f"_{which_results}_optimized_results", dict_results, allow_pickle=True)
    if which_results == "main":
        np.save(directory_save+"_best_params", np.array(best_params))


def compare_human_vertex_models():
    pass


# Current working director
cwd = os.getcwd()

# Global variable here. So it only is defined once. 
path_data = f"{cwd}/data/human_high_res"

# Path to the Conda environment
conda_prefix = os.environ.get("CONDA_PREFIX")

env_path = conda_prefix.split("/conda/")[0]
# print(env_path, "base_conda_path")

# Current environment name
conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
# print(conda_env_name)

# sys.exit()

def mainFunction():
    """
    Main function to reproduce the analysis on the high-resolution human connectome.

    These functions have to be run sequentially.
    """

    
    # 1. Generate the geometric eigenmodes
    generate_geometric_modes()

    # 2. Optimize the GEM (explore parameters landscape)
    # optimize_and_save_human_high_resolution_results()
    # print()


    #3. Visualize performance
    # visualize_GEM_human_vertex_results()

    #4. Generate benchmark models
    # results = "main"
    # results = "modularity"
    # results = "spectral"
    # generate_human_vertex_comparison_results(which_results=results)

    #5. Compare GEM performance with other models
    # visualize_human_vertex_models_comparison()


if __name__ == "__main__":
    mainFunction()