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

from utilities import powerlawRule, exponentialRule

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


def EDR_generate_and_save(species, dense_or_sparse, path_data, task_id):
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
    eta_range_id = task_id // 10 # 0 to 99 (etas)
    repetition_id = task_id % 10 # 0 to 9 (repetitions)

    formulation = "EDR-vertex"
    # number_of_repetitions = 10 # All repetitions are saved separately to make it more efficient.

    from demo_non_human_species import get_animal_paramameters, get_non_human_species_mesh_and_empirical_connectome, get_non_human_species_EDR_parameters

    species_parameters = get_animal_paramameters(species, dense_or_sparse)

    surface_name, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights = species_parameters
    
    loaded_parameters_and_variables = np.load(path_data + f"/{species}/{species}_parc_scheme={mean_or_sum}_saved_parameters_and_variables.npy", allow_pickle=True).item()
    idxes_cortex = loaded_parameters_and_variables['idxes_cortex']
    vertices_in_connectome = loaded_parameters_and_variables['vertices_in_connectome']
    
    model_parameters_and_variables = {}
    model_parameters_and_variables["target_density"] = target_density
    model_parameters_and_variables["species"] = species
    model_parameters_and_variables["density_allen"] = density_allen
    model_parameters_and_variables['fixed_threshold_vertex'] = fixed_threshold_vertex

    model_parameters_and_variables["characteristic_matrix"] = loaded_parameters_and_variables['characteristic_matrix']
    model_parameters_and_variables['vertices_in_connectome'] = vertices_in_connectome
    model_parameters_and_variables['idxes_cortex'] = idxes_cortex
    
    mesh, empirical_parcel_connectivity, n_edges_empirical_parcel = get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables)
    
    model_parameters_and_variables['n_edges_empirical_parcel'] = n_edges_empirical_parcel

    vertices = mesh.v

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

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "clustering", "node connection distance"]

    print(f"target_density: {target_density}")
    print("formulation:", formulation)
    
    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)
    model_parameters_and_variables['idxes_parcel'] = idxes_parcel

    total_possible_connections = len(idxes_vertex[0])
    model_parameters_and_variables['total_possible_connections'] = total_possible_connections

    n_connections_vertex = int(fixed_threshold_vertex * total_possible_connections)
    model_parameters_and_variables['n_connections_vertex'] = n_connections_vertex

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    distances, centroids = utilities.get_non_human_species_centroids(species, path_data)

    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)

    eta_prob_connection_array, eta_prob_connection_array_split = get_non_human_species_EDR_parameters()

    eta_prob_connection_range = eta_prob_connection_array_split[eta_range_id]

    current_hypothesis = f"formulation={formulation}_target_density={target_density}_eta_range_id_{eta_range_id}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/non_human_species/{species}/resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    ##################################### CHECKING IF ALREADY EXISTS
    exits = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exits == True:
        print(exits, "exits")
        return True #Skipping
    ##################################### 

    results_dict = {network_metric:np.empty(len(eta_prob_connection_range)) for network_metric in network_measures}
    for idx_eta_prob, eta_prob_connection in enumerate(eta_prob_connection_range):
        
        modelSC = connectome_models.generate_EDR_non_human_species_model(eta_prob_connection, model_parameters_and_variables)
        modelSC_idxes = modelSC[idxes_parcel]
        idxes_edges_model = np.nonzero(modelSC_idxes)[0]        
        if len(network_measures) != 0:
            compute_and_update_results(results_dict, idx_eta_prob, network_measures, modelSC, modelSC_idxes, empirical_node_properties_dict,  empirical_parcel_connectivity_idxes, idxes_edges_empirical, distances)
        # print(time.time() - start_time, "time for one eta")

    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")

def distance_atlas_generate_and_save(species, dense_or_sparse, path_data, repetition_id):

    dense_or_sparse = "sparse"
    
    formulation = "distance-atlas"

    rule="powerlaw"
    # rule="exponential"

    if rule=="exponential":
        cost_rule = exponentialRule
        print("exponential")
    elif rule=="powerlaw":
        cost_rule = powerlawRule
        print("powerlaw")

    from demo_non_human_species import get_animal_paramameters, get_non_human_species_mesh_and_empirical_connectome, get_non_human_species_distance_atlas_parameters

    species_parameters = get_animal_paramameters(species, dense_or_sparse)

    _, density_allen, _, _, target_density, _, resampling_weights = species_parameters

    model_parameters_and_variables = {}
    model_parameters_and_variables["target_density"] = target_density
    model_parameters_and_variables["species"] = species
    model_parameters_and_variables["density_allen"] = density_allen

    _, empirical_parcel_connectivity, n_edges_empirical_parcel = get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables)

    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity = (empirical_parcel_connectivity > 0).astype(int)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    distances, centroids = utilities.get_non_human_species_centroids(species, path_data)
    distances /= np.max(distances)

    current_hypothesis = f"formulation={formulation}_target_density={target_density}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/non_human_species/{species}/resampled_weights_{resampling_weights}_formulation_{formulation}"
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

    eta = get_non_human_species_distance_atlas_parameters(species)

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

def matching_index_generate_and_save(species, dense_or_sparse, path_data, repetition_id):

    dense_or_sparse = "sparse"

    formulation = "MI"

    rule="powerlaw"
    # rule="exponential"

    if rule=="exponential":
        cost_rule = exponentialRule
        print("exponential")
    elif rule=="powerlaw":
        cost_rule = powerlawRule
        print("powerlaw")

    from demo_non_human_species import get_animal_paramameters, get_non_human_species_mesh_and_empirical_connectome, get_non_human_species_matching_index_parameters

    species_parameters = get_animal_paramameters(species, dense_or_sparse)

    _, density_allen, _, _, target_density, _, resampling_weights = species_parameters

    model_parameters_and_variables = {}
    model_parameters_and_variables["target_density"] = target_density
    model_parameters_and_variables["species"] = species
    model_parameters_and_variables["density_allen"] = density_allen

    _, empirical_parcel_connectivity, n_edges_empirical_parcel = get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables)

    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity = (empirical_parcel_connectivity > 0).astype(int)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    distances, centroids = utilities.get_non_human_species_centroids(species, path_data)
    distances /= np.max(distances)

    current_hypothesis = f"formulation={formulation}_target_density={target_density}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/non_human_species/{species}/resampled_weights_{resampling_weights}_formulation_{formulation}"
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

    eta, gamma = get_non_human_species_matching_index_parameters(species)
    
    results_dict = {net_measure:np.empty((len(gamma), len(eta))) for net_measure in network_measures}

    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)
    total_number_of_possible_edges = len(idxes_parcel[0])

    for idx_gamma, gamma_i in enumerate(gamma):
        
        for idx_eta, eta_i in enumerate(eta):

            modelSC = connectome_models.generate_matching_index_model(eta_i, gamma_i, n_nodes, n_edges_parcel_empirical, distances, total_number_of_possible_edges, idxes_parcel, cost_rule)
            modelSC_idxes = modelSC[idxes_parcel]

            if len(network_measures) != 0:
                compute_and_update_results(results_dict, idx_eta, network_measures, modelSC, modelSC_idxes, empirical_node_properties_dict,  empirical_parcel_connectivity_idxes, idxes_edges_empirical, distances, idx_gamma=idx_gamma)
    
    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")


def generate_and_save_model_performance(species, dense_or_sparse, path_data, r_s_id=None, formulation="GEM"):

    if formulation == "EDR-vertex":
        return EDR_generate_and_save(species, dense_or_sparse, path_data, task_id=r_s_id)

    elif formulation == "distance-atlas":
        return distance_atlas_generate_and_save(species, dense_or_sparse, path_data, repetition_id=r_s_id)

    elif formulation == "MI":
        return matching_index_generate_and_save(species, dense_or_sparse, path_data, repetition_id=r_s_id)
    
    from demo_non_human_species import get_animal_paramameters, get_non_human_species_mesh_and_empirical_connectome, load_non_human_species_modes

    lump = False ## Fixed. Will override previous files if changed.

    species_parameters = get_animal_paramameters(species, dense_or_sparse)
    surface_name, density_allen, mean_or_sum, r_s_values_list, target_density, fixed_threshold_vertex, resampling_weights = species_parameters
    
    loaded_parameters_and_variables = np.load(path_data + f"/{species}/{species}_parc_scheme={mean_or_sum}_saved_parameters_and_variables.npy", allow_pickle=True).item()
    mesh, _ = utilities.get_non_human_species_mesh(path_data, species)
    
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
    model_parameters_and_variables['cortex_mask'] = cortex_mask
    model_parameters_and_variables["species"] = species
    model_parameters_and_variables["density_allen"] = density_allen
    model_parameters_and_variables['fixed_threshold_vertex'] = fixed_threshold_vertex

    model_parameters_and_variables["characteristic_matrix"] = loaded_parameters_and_variables['characteristic_matrix']
    model_parameters_and_variables['vertices_in_connectome'] = loaded_parameters_and_variables["vertices_in_connectome"]
    model_parameters_and_variables['idxes_cortex'] = idxes_cortex
    
    _, empirical_parcel_connectivity, n_edges_empirical_parcel = get_non_human_species_mesh_and_empirical_connectome(model_parameters_and_variables)
    
    model_parameters_and_variables['n_edges_empirical_parcel'] = n_edges_empirical_parcel


    evals, emodes = load_non_human_species_modes(species, lump=False)
    
    k_range = np.array([k_ for k_ in range(2, 200)])

    emodes = emodes[:, 0:k_range.max()+1]
    evals = evals[0:k_range.max()+1]

    model_parameters_and_variables["evals"] = evals
    model_parameters_and_variables["emodes"] = emodes

    if formulation == "GEM":
        network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "clustering", "node connection distance"]
    else:
        network_measures = ["true_positive_rate", "degreeBinary", "clustering", "node connection distance"]

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
    
    path_base_save = f"/{cwd}/data/results/non_human_species/{species}/resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    ##################################### CHECKING IF ALREADY EXISTS
    exists = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exists == True:
        print("files already exists")
        return True
    ##################################### 
    
    n_nodes = empirical_parcel_connectivity.shape[0]
    idxes_parcel = np.triu_indices(n_nodes, k=1)

    empirical_parcel_connectivity_idxes = empirical_parcel_connectivity[idxes_parcel]
    idxes_edges_empirical = np.nonzero(empirical_parcel_connectivity_idxes)[0]
    n_edges_parcel_empirical = len(idxes_edges_empirical)
    density = n_edges_parcel_empirical/len(idxes_parcel[0])

    print(n_edges_parcel_empirical, "n_edges_parcel_empirical")

    distances, centroids = utilities.get_non_human_species_centroids(species, path_data)

    distances_idxes_edges_empirical = distances[idxes_edges_empirical]

    results_dict = {network_metric:np.empty(len(k_range)) for network_metric in network_measures}
        
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_parcel_connectivity, distances)

    for k_idx, k in enumerate(k_range):
        print(k, "k")
        modelSC = connectome_models.generate_non_human_species_GEM(r_s, k, model_parameters_and_variables, resampling_weights=resampling_weights)
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
    parser.add_argument("--species", required=False, default=None)
    parser.add_argument("--dense_or_sparse", required=False, default=None) 
    args = parser.parse_args()
    
    path_data = args.path_data
    r_s_id = args.r_s_id
    formulation = args.formulation
    species = args.species
    dense_or_sparse = args.dense_or_sparse

    if r_s_id is None:
        r_s_id = 0

    else:
        r_s_id = int(r_s_id)

    if path_data is None:
        path_data = f"/{cwd}/data/non_human_species"

    print(path_data, "pathj data wtf")

    if formulation is None:
        # Manual input here instead of call from command line 
        
        formulation = "GEM"
        # formulation = "EDR-vertex"
        # formulation = "distance-atlas"
        # formulation = "MI"

    if species == None:
        # species = "Mouse"
        species = "Marmoset"
        # species = "Macaque"

    if dense_or_sparse == None:
        dense_or_sparse = "dense"
        # dense_or_sparse = "sparse"
        
    generate_and_save_model_performance(species, dense_or_sparse, path_data, r_s_id, formulation)

    os._exit(0)