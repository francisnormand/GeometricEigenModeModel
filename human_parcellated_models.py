import numpy as np
import os
import sys
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr, rankdata
import argparse
from numba import njit, jit
from scipy.stats import ks_2samp
from network_measures_and_statistics import compute_node_properties, compute_true_positive_rate, fast_spearmanr_numba, fast_pearsonr
import connectome_models
import utilities
import time

cwd = os.getcwd()

def check_if_already_exists(network_measures, path_base_save, current_hypothesis):
    missing_measures = []
    already_exists = []
    for network_meaz in network_measures:
        if not os.path.exists(path_base_save + f"/{network_meaz}_{current_hypothesis}.npy"):
            missing_measures.append(network_meaz)
        else:
            already_exists.append(network_meaz)

    if len(already_exists) == len(network_measures):
        print()
        print(f"{network_measures} all exists already")
        print()
        return True
    
    else:
        network_measures = [missing_measures[i] for i in range(len(missing_measures))]    
    
    print(network_measures, "network measures to be computed")
    print()
    return False


def compute_and_update_results(results_dict, k_idx, network_measures, vertexModelSC, vertexModelSC_idxes, empirical_node_properties_dict, empirical_vertex_connectivity_idxes, idxes_edges_empirical, distances, idx_gamma=False):

    node_properties_model_dict = compute_node_properties(network_measures, vertexModelSC, distances)
    for measure_ in network_measures:
        if measure_ == "spearman_union_weights":
            idxes_union = np.where((vertexModelSC_idxes != 0) | (empirical_vertex_connectivity_idxes != 0))[0]
            if idx_gamma == False:
                results_dict[measure_][k_idx] = fast_spearmanr_numba(vertexModelSC_idxes[idxes_union], empirical_vertex_connectivity_idxes[idxes_union])
            else:
                results_dict[measure_][idx_gamma, k_idx] = fast_spearmanr_numba(vertexModelSC_idxes[idxes_union], empirical_vertex_connectivity_idxes[idxes_union])
        
        elif measure_ == "true_positive_rate":
            if idx_gamma == False:
                results_dict[measure_][k_idx] = compute_true_positive_rate(vertexModelSC_idxes, idxes_edges_empirical)
            else:
                results_dict[measure_][idx_gamma, k_idx] = compute_true_positive_rate(vertexModelSC_idxes, idxes_edges_empirical)
        else:
            if idx_gamma == False:
                results_dict[measure_][k_idx] = spearmanr(node_properties_model_dict[measure_], empirical_node_properties_dict[measure_])[0]
            else:
                results_dict[measure_][idx_gamma, k_idx] = spearmanr(node_properties_model_dict[measure_], empirical_node_properties_dict[measure_])[0]


def EDR_generate_and_save(path_data, task_id, number_of_parcels=300):
    """
    Generate and save EDR model performance for a given parameter set.

    Parameters
    ----------
    path_data : str or Path
        Path to the data directory.
    task_id : int
        Task index encoding both parameter (eta_id) and repetition (repetition_id).

    Notes
    -----
    - eta_id is derived as task_id // 10 representing the $\eta_p$ parameter in the paper (0–99)
    - repetition_id is derived as task_id % 10  (0–9)
    """
    eta_id = task_id // 10 # 0 to 99 (etas)
    repetition_id = task_id % 10 # 0 to 9 (repetitions)

    formulation = "EDR-vertex"
    # number_of_repetitions = 10 # All repetitions are saved separately to make it more efficient.

    from demo_human_parcellated import get_human_parcellated_parameters, get_human_high_res_surface_and_parcellated_connectome, get_human_parcellated_EDR_parameters

    human_parcellated_parameters = get_human_parcellated_parameters(number_of_parcels)
    _, cortex_mask, connectome_type, fwhm, target_density, fixed_vertex_threshold_density, resampling_weights = human_parcellated_parameters

    (surface, _), cortex_mask_array, empirical_parcel_connectivity = get_human_high_res_surface_and_parcellated_connectome(path_data, number_of_parcels, human_parcellated_parameters)

    vertices = surface.v

    if cortex_mask == True:
        idxes_cortex = np.where(cortex_mask_array == 1)[0]
        vertices = vertices[idxes_cortex, :]

    distances_vertices = pdist(vertices)
    n_vertices = vertices.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "clustering", "node connection distance"]

    print(f"target_density: {target_density}")
    print("formulation:", formulation)
    
    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    total_possible_connections = len(idxes_vertex[0])

    n_edges_vertex_empirical = int(fixed_vertex_threshold_density * total_possible_connections)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    distances, centroids = utilities.get_parcellated_human_centroids(number_of_parcels)
    distances_idxes_edges_empirical = distances[idxes_edges_empirical]

    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)

    eta_prob_connection_array, eta_weights_array = get_human_parcellated_EDR_parameters(number_of_parcels)

    eta_prob_connection = eta_prob_connection_array[eta_id]

    if connectome_type == "smoothed":
        current_hypothesis = f"formulation={formulation}_fwhm={fwhm}_target_density={target_density}_eta_prob_conn_id_{eta_id}_repetition_id_{repetition_id}"
    else:
        current_hypothesis = f"formulation={formulation}_target_density={target_density}_eta_prob_conn_id_{eta_id}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/human_parcellated/Schaefer{number_of_parcels}/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    ##################################### CHECKING IF ALREADY EXISTS
    exits = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exits == True:
        print(exits, "exits")
        return True #Skipping
    ##################################### 

    results_dict = {network_metric:np.empty(len(eta_weights_array)) for network_metric in network_measures}
    characteristic_matrix = np.load(f"{path_data}/Schaefer{number_of_parcels}/characteristic_matrix_to_SC{number_of_parcels}.npy")

    for idx_eta_w, eta_w in enumerate(eta_weights_array):
        modelSC = connectome_models.generate_EDR_vertex_parcellated_model(eta_prob_connection, eta_w, distances_vertices, idxes_vertex, idxes_parcel, n_vertices, n_edges_vertex_empirical, total_possible_connections, resampling_weights, characteristic_matrix, n_edges_parcel_empirical)
    
        modelSC_idxes = modelSC[idxes_parcel]
        idxes_edges_model = np.nonzero(modelSC_idxes)[0]        
        if len(network_measures) != 0:
            compute_and_update_results(results_dict, idx_eta_w, network_measures, modelSC, modelSC_idxes, empirical_node_properties_dict,  empirical_parcel_connectivity_idxes, idxes_edges_empirical, distances)

        # print(time.time() - start_time, "time for one eta")

    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")

def powerlawRule(ditanceArray, eta):
    return ditanceArray**eta

def exponentialRule(ditanceArray, eta):
    return np.exp(eta*ditanceArray)

def distance_atlas_generate_and_save(path_data, number_of_parcels, repetition_id):

    formulation = "distance-atlas"

    rule="powerlaw"
    # rule="exponential"

    if rule=="exponential":
        cost_rule = exponentialRule
        print("exponential")
    elif rule=="powerlaw":
        cost_rule = powerlawRule
        print("powerlaw")

    from demo_human_parcellated import get_human_parcellated_parameters, get_human_high_res_surface_and_parcellated_connectome, get_human_parcellated_EDR_parameters

    human_parcellated_parameters = get_human_parcellated_parameters(number_of_parcels)
    _, _, connectome_type, fwhm, target_density, _, resampling_weights = human_parcellated_parameters

    (_, _), _, empirical_parcel_connectivity = get_human_high_res_surface_and_parcellated_connectome(path_data, number_of_parcels, human_parcellated_parameters)

    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity = (empirical_parcel_connectivity > 0).astype(int)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    distances, centroids = utilities.get_parcellated_human_centroids(number_of_parcels)
    distances /= np.max(distances)

    if connectome_type == "smoothed":
        current_hypothesis = f"formulation={formulation}_fwhm={fwhm}_target_density={target_density}_repetition_id_{repetition_id}"
    else:
        current_hypothesis = f"formulation={formulation}_target_density={target_density}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/human_parcellated/Schaefer{number_of_parcels}/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    network_measures = ["true_positive_rate", "degreeBinary", "clustering", "node connection distance"]

    print(f"target_density: {target_density}")
    print("formulation:", formulation)

    ##################################### CHECKING IF ALREADY EXISTS
    exits = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exits == True:
        print(exits, "exits")
        return True #Skipping
    ##################################### 

    eta = np.linspace(-11, -3, 10000)
    results_dict = {net_measure:np.empty(len(eta)) for net_measure in network_measures}

    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)

    for idx_eta, eta_i in enumerate(eta):
        
        modelSC = connectome_models.generate_distance_atlas_model(eta_i, n_nodes, n_edges_parcel_empirical, distances, idxes_parcel, cost_rule)
        modelSC_idxes = modelSC[idxes_parcel]

        if len(network_measures) != 0:
            compute_and_update_results(results_dict, idx_eta, network_measures, modelSC, modelSC_idxes, empirical_node_properties_dict,  empirical_parcel_connectivity_idxes, idxes_edges_empirical, distances)
    
    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")


def matching_index_generate_and_save(path_data, number_of_parcels, repetition_id):

    formulation = "MI"

    rule="powerlaw"
    # rule="exponential"

    if rule=="exponential":
        cost_rule = exponentialRule
        print("exponential")
    elif rule=="powerlaw":
        cost_rule = powerlawRule
        print("powerlaw")

    from demo_human_parcellated import get_human_parcellated_parameters, get_human_high_res_surface_and_parcellated_connectome, get_human_parcellated_EDR_parameters

    human_parcellated_parameters = get_human_parcellated_parameters(number_of_parcels)
    _, _, connectome_type, fwhm, target_density, _, resampling_weights = human_parcellated_parameters

    (_, _), _, empirical_parcel_connectivity = get_human_high_res_surface_and_parcellated_connectome(path_data, number_of_parcels, human_parcellated_parameters)

    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity = (empirical_parcel_connectivity > 0).astype(int)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    distances, centroids = utilities.get_parcellated_human_centroids(number_of_parcels)
    distances /= np.max(distances)

    if connectome_type == "smoothed":
        current_hypothesis = f"formulation={formulation}_fwhm={fwhm}_target_density={target_density}_repetition_id_{repetition_id}"
    else:
        current_hypothesis = f"formulation={formulation}_target_density={target_density}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/human_parcellated/Schaefer{number_of_parcels}/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    network_measures = ["true_positive_rate", "degreeBinary", "clustering", "node connection distance"]

    print(f"target_density: {target_density}")
    print("formulation:", formulation)

    ##################################### CHECKING IF ALREADY EXISTS
    exits = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exits == True:
        print(exits, "exits")
        return True #Skipping
    ##################################### 

    eta = np.linspace(-11, -3, 100)
    gamma = np.linspace(0.1, 0.7, 100)
    
    results_dict = {net_measure:np.empty((len(gamma), len(eta))) for net_measure in network_measures}

    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)
    total_number_of_possible_edges = len(idxes_parcel[0])

    # print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    for idx_gamma, gamma_i in enumerate(gamma):
        
        for idx_eta, eta_i in enumerate(eta):
            # print(idx_eta, "idx eta")

            modelSC = connectome_models.generate_matching_index_model(eta_i, gamma_i, n_nodes, n_edges_parcel_empirical, distances, total_number_of_possible_edges, idxes_parcel, cost_rule)
            modelSC_idxes = modelSC[idxes_parcel]

            if len(network_measures) != 0:
                compute_and_update_results(results_dict, idx_eta, network_measures, modelSC, modelSC_idxes, empirical_node_properties_dict,  empirical_parcel_connectivity_idxes, idxes_edges_empirical, distances, idx_gamma=idx_gamma)
    
    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")


def generate_and_save_model_performance(number_of_parcels, path_data, r_s_id=None, formulation="GEM"):

    if formulation == "EDR-vertex":
        return EDR_generate_and_save(path_data, task_id=r_s_id, number_of_parcels=number_of_parcels)

    elif formulation == "distance-atlas":
        return distance_atlas_generate_and_save(path_data, number_of_parcels=number_of_parcels, repetition_id=r_s_id)

    elif formulation == "MI":
        return matching_index_generate_and_save(path_data, number_of_parcels=number_of_parcels, repetition_id=r_s_id)
    
    from demo_human_parcellated import get_human_parcellated_parameters, get_human_high_res_surface_and_parcellated_connectome, load_human_parcellated_modes

    lump = False ## Fixed. Will override previous files if changed.

    human_parcellated_parameters = get_human_parcellated_parameters(number_of_parcels)
    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, fixed_vertex_threshold_density, resampling_weights = human_parcellated_parameters

    (_, _), cortex_mask_array, empirical_parcel_connectivity = get_human_high_res_surface_and_parcellated_connectome(path_data, number_of_parcels, human_parcellated_parameters)
    
    evals, emodes = load_human_parcellated_modes(path_data, number_of_parcels, lump, cortex_mask)

    k_range = np.array([k_ for k_ in range(2, 200)])

    if formulation == "GEM":
        network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "clustering", "node connection distance"]
    else:
        network_measures = ["true_positive_rate", "degreeBinary", "clustering", "node connection distance"]

    if connectome_type == "smoothed":
        current_hypothesis = f"formulation={formulation}_fwhm={fwhm}_target_density={target_density}"
    else:
        current_hypothesis = f"formulation={formulation}_target_density={target_density}"

    if formulation == "GEM":
        current_hypothesis += f"_r_s_id={r_s_id}"
        r_s = r_s_values_list[r_s_id]
    else:
        r_s = None

    print(f"current_hypothesis :{current_hypothesis}")
    print(f"target_density: {target_density}")
    print(f"r_s: {r_s}")
    print("formulation:", formulation)
    
    path_base_save = f"/{cwd}/data/results/human_parcellated/Schaefer{number_of_parcels}/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    ##################################### CHECKING IF ALREADY EXISTS
    exists = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exists == True:
        return True
    ##################################### 
    
    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    if cortex_mask == True:
        idxes_cortex = np.where(cortex_mask_array == 1)[0]
        emodes = emodes[idxes_cortex, :]
    
    emodes = emodes[:, 0:k_range.max()+1]
    evals = evals[0:k_range.max()+1]
    n_vertices = emodes.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)

    distances, centroids = utilities.get_parcellated_human_centroids(number_of_parcels)
    distances_idxes_edges_empirical = distances[idxes_edges_empirical]

    results_dict = {network_metric:np.empty(len(k_range)) for network_metric in network_measures}
        
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)

    characteristic_matrix = np.load(f"{path_data}/Schaefer{number_of_parcels}/characteristic_matrix_to_SC{number_of_parcels}.npy")

    for k_idx, k in enumerate(k_range):
        
        modelSC = connectome_models.generate_parcellated_GEM_humans(r_s, k, emodes, evals, idxes_vertex, idxes_parcel, characteristic_matrix, fixed_vertex_threshold_density, n_edges_parcel_empirical, resampling_weights)
        modelSC_idxes = modelSC[idxes_parcel]
        idxes_edges_model = np.nonzero(modelSC_idxes)[0]        
        
        if len(network_measures) != 0:
            compute_and_update_results(results_dict, k_idx, network_measures, modelSC, modelSC_idxes, empirical_node_properties_dict,  empirical_parcel_connectivity_idxes, idxes_edges_empirical, distances)

    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", required=False, default=None)
    parser.add_argument("--r_s_id", required=False, default=None)
    parser.add_argument("--formulation", required=False, default=None)
    
    args = parser.parse_args()
    
    path_data = args.path_data
    r_s_id = args.r_s_id
    formulation = args.formulation

    number_of_parcels = 300

    if r_s_id is None:
        r_s_id = 0

    else:
        r_s_id = int(r_s_id)

    if path_data is None:
        path_data = f"/{cwd}/data/human_parcellated"

    if formulation is None:
        # Manual input here instead of call from command line 
        # formulation = "GEM"
        # formulation = "EDR-vertex"
        # formulation = "distance-atlas"
        formulation = "MI"
        
    generate_and_save_model_performance(number_of_parcels, path_data, r_s_id, formulation)

    os._exit(0)

