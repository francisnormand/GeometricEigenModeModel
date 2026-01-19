import numpy as np
import os
import sys
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr, rankdata
import argparse
from numba import njit, jit
from scipy.stats import ks_2samp
from network_measures_and_statistics import compute_node_properties
import connectome_models
import utilities
import time

cwd = os.getcwd()

def compute_true_positive_rate(vertexSpaceSC_thresholded_binary_idxes, idxes_edges_empirical):
    return len(np.where(vertexSpaceSC_thresholded_binary_idxes[idxes_edges_empirical] != 0)[0])/len(idxes_edges_empirical)

@njit(parallel=True)
def rankdata_numba(a):
    n = len(a)
    ivec = np.argsort(a)
    svec = np.empty(n, dtype=np.int64)
    svec[ivec] = np.arange(n)
    rvec = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and a[ivec[j]] == a[ivec[j + 1]]:
            j += 1
        rank = 0.5 * (i + j + 1)
        for k in range(i, j + 1):
            rvec[ivec[k]] = rank
        i = j + 1
    return rvec

@jit(nopython=True, fastmath=True)
def fast_spearmanr_numba(x, y):
    x_ranked = rankdata_numba(x)
    y_ranked = rankdata_numba(y)
    
    return fast_pearsonr(x_ranked, y_ranked)


@njit(parallel=True)
def fast_pearsonr(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    
    return numerator / denominator


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


def compute_and_update_results(results_dict, k_idx, network_measures, vertexModelSC, vertexModelSC_idxes, empirical_node_properties_dict, empirical_vertex_connectivity_idxes, idxes_edges_empirical, distances):

    node_properties_model_dict = compute_node_properties(network_measures, vertexModelSC, distances)
    for measure_ in network_measures:
        if measure_ == "spearman_union_weights":
            idxes_union = np.where((vertexModelSC_idxes != 0) | (empirical_vertex_connectivity_idxes != 0))[0]
            
            results_dict[measure_][k_idx] = fast_spearmanr_numba(vertexModelSC_idxes[idxes_union], empirical_vertex_connectivity_idxes[idxes_union])
        
        elif measure_ == "true_positive_rate":
            results_dict[measure_][k_idx] = compute_true_positive_rate(vertexModelSC_idxes, idxes_edges_empirical)
        
        else:
            results_dict[measure_][k_idx] = spearmanr(node_properties_model_dict[measure_], empirical_node_properties_dict[measure_])[0]


def get_human_vertex_results(network_measures, vertexModelSC, vertexModelSC_thresholded_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances):

    node_properties_model_dict = compute_node_properties(network_measures, vertexModelSC, distances)

    results_dict = {net_measure:0 for net_measure in network_measures}
    
    idxes_edges_empirical = np.nonzero(empirical_vertex_connectivity_idxes)[0]

    vertexModelSC_binary = (vertexModelSC > 0).astype(int)

    idxes_vertex = np.triu_indices(vertexModelSC.shape[0], k=1)

    empirical_connectome_binary_idxes = (empirical_vertex_connectivity_idxes > 0).astype(int)

    for measure_ in network_measures:
        if measure_ == "spearman_union_weights":
            idxes_union = np.where((vertexModelSC_thresholded_idxes != 0) | (empirical_vertex_connectivity_idxes != 0))[0]
            results_dict[measure_] = fast_spearmanr_numba(vertexModelSC_thresholded_idxes[idxes_union], empirical_vertex_connectivity_idxes[idxes_union])
        
        elif measure_ == "true_positive_rate":
            results_dict[measure_] = compute_true_positive_rate(vertexModelSC_thresholded_idxes, idxes_edges_empirical)
        
        else:    
            results_dict[measure_] = spearmanr(node_properties_model_dict[measure_], empirical_node_properties_dict[measure_])[0]

    return results_dict

def EDR_generate_and_save(path_data, task_id):
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

    formulation = "EDR"
    # number_of_repetitions = 10 # All repetitions are saved separately to make it more efficient.

    from demo_high_resolution import get_human_vertex_parameters, get_human_high_res_surface_and_connectome, load_human_vertex_modes, get_human_vertex_EDR_parameters

    human_vertex_parameters = get_human_vertex_parameters()
    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, resampling_weights = human_vertex_parameters

    (surface, _), cortex_mask_array, empirical_vertex_connectivity = get_human_high_res_surface_and_connectome(path_data, human_vertex_parameters)
    
    vertices = surface.v

    if cortex_mask == True:
        idxes_cortex = np.where(cortex_mask_array == 1)[0]
        vertices = vertices[idxes_cortex, :]

    distances = pdist(vertices)

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "clustering", "node connection distance"]
    distance_measures = []

    print(f"target_density: {target_density}")
    print("formulation:", formulation)
    
    n_vertices = empirical_vertex_connectivity.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)

    total_possible_connections = len(idxes_vertex[0])

    empirical_vertex_connectivity_idxes = empirical_vertex_connectivity[idxes_vertex]
    idxes_edges_empirical = np.nonzero(empirical_vertex_connectivity_idxes)[0]
    n_edges_vertex_empirical = len(idxes_edges_empirical)
    density = n_edges_vertex_empirical/len(idxes_vertex[0])

    distances_idxes_edges_empirical = distances[idxes_edges_empirical]

    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_vertex_connectivity, distances)

    eta_prob_connection_array, eta_weights_array = get_human_vertex_EDR_parameters()

    eta_prob_connection = eta_prob_connection_array[eta_id]

    if connectome_type == "smoothed":
        current_hypothesis = f"formulation={formulation}_fwhm={fwhm}_target_density={target_density}_eta_prob_conn_id_{eta_id}_repetition_id_{repetition_id}"
    else:
        current_hypothesis = f"formulation={formulation}_target_density={target_density}_eta_prob_conn_id_{eta_id}_repetition_id_{repetition_id}"

    print(f"current_hypothesis :{current_hypothesis}")

    path_base_save = f"/{cwd}/data/results/human_high_resolution/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    ##################################### CHECKING IF ALREADY EXISTS
    exits = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exits == True:
        print(exits, "exits")
        return True #Skipping
    ##################################### 

    results_dict = {network_metric:np.empty(len(eta_weights_array)) for network_metric in network_measures}

    for idx_eta_w, eta_w in enumerate(eta_weights_array):
        vertexModelSC = connectome_models.generate_EDR_vertex_model(eta_prob_connection, eta_w, distances, idxes_vertex, n_vertices, n_edges_vertex_empirical, total_possible_connections, resampling_weights)

        vertexModelSC_idxes = vertexModelSC[idxes_vertex]
        idxes_edges_model = np.nonzero(vertexModelSC_idxes)[0]        

        if len(network_measures) != 0:
            compute_and_update_results(results_dict, idx_eta_w, network_measures, vertexModelSC, vertexModelSC_idxes, empirical_node_properties_dict,  empirical_vertex_connectivity_idxes, idxes_edges_empirical, distances)

        # print(time.time() - start_time, "time for one eta")

    for net_measure in results_dict.keys():
        np.save(path_base_save + f"/{net_measure}_{current_hypothesis}", results_dict[net_measure])

    print(f"done and saved {formulation}")

def generate_and_save_model_performance(path_data, r_s_id=None, formulation="GEM"):

    if formulation == "EDR":
        return EDR_generate_and_save(path_data, task_id=r_s_id)

    from demo_high_resolution import get_human_vertex_parameters, get_human_high_res_surface_and_connectome, load_human_vertex_modes

    if formulation == "GEM":
        connectome_model_used = connectome_models.generate_high_res_GEM_humans
    elif formulation == "LBO":
        connectome_model_used = connectome_models.generate_high_res_LBO_humans

    lump = False ## Fixed. Will override previous files if changed.

    human_vertex_parameters = get_human_vertex_parameters()
    r_s_values_list, cortex_mask, connectome_type, fwhm, target_density, resampling_weights = human_vertex_parameters

    (surface, _), cortex_mask_array, empirical_vertex_connectivity = get_human_high_res_surface_and_connectome(path_data, human_vertex_parameters)
    
    evals, emodes = load_human_vertex_modes(path_data, lump, cortex_mask)

    vertices = surface.v

    k_range = np.array([k_ for k_ in range(2, 200)])

    network_measures = ["degree", "true_positive_rate", "degreeBinary", "spearman_union_weights", "ranked_weights_strength", "clustering", "node connection distance"]

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
    
    path_base_save = f"/{cwd}/data/results/human_high_resolution/{connectome_type}_resampled_weights_{resampling_weights}_formulation_{formulation}"
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    ##################################### CHECKING IF ALREADY EXISTS
    exists = check_if_already_exists(network_measures, path_base_save, current_hypothesis)
    if exists == True:
        return True
    ##################################### 
    
    n_vertices = empirical_vertex_connectivity.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)

    empirical_vertex_connectivity_idxes = empirical_vertex_connectivity[idxes_vertex]
    idxes_edges_empirical = np.nonzero(empirical_vertex_connectivity_idxes)[0]
    n_edges_vertex_empirical = len(idxes_edges_empirical)
    density = n_edges_vertex_empirical/len(idxes_vertex[0])

    if cortex_mask == True:
        idxes_cortex = np.where(cortex_mask_array == 1)[0]
        emodes = emodes[idxes_cortex, :]
        vertices = vertices[idxes_cortex, :]
    
    emodes = emodes[:, 0:k_range.max()+1]
    evals = evals[0:k_range.max()+1]

    distances = pdist(vertices)
    distances_idxes_edges_empirical = distances[idxes_edges_empirical]

    results_dict = {network_metric:np.empty(len(k_range)) for network_metric in network_measures}
        
    empirical_node_properties_dict = compute_node_properties(network_measures, empirical_vertex_connectivity, distances)
    
    for k_idx, k in enumerate(k_range):
    
        vertexModelSC = connectome_model_used(r_s, k, emodes, evals, density, idxes_vertex, resampling_weights)
        vertexModelSC_idxes = vertexModelSC[idxes_vertex]
        idxes_edges_model = np.nonzero(vertexModelSC_idxes)[0]        

        if len(network_measures) != 0:
            compute_and_update_results(results_dict, k_idx, network_measures, vertexModelSC, vertexModelSC_idxes, empirical_node_properties_dict,  empirical_vertex_connectivity_idxes, idxes_edges_empirical, distances)

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

    if r_s_id is None:
        r_s_id = 0

    else:
        r_s_id = int(r_s_id)

    if path_data is None:
        path_data = f"/{cwd}/data/human_high_res"

    if formulation is None:
        # Manual input here instead of call from command line 
        formulation = "EDR-vertex"
        # formulation = "distance-atlas"
        # formulation = "MI"
        # formulation = "GEM"

    generate_and_save_model_performance(path_data, r_s_id, formulation)

    os._exit(0)

