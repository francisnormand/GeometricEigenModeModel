import utilities
import numpy as np



def generate_high_res_GEM_humans(r_s, k, emodes, evals, density, idxes_vertex, resampling_weights):
    
    evals_r_s_squared = (r_s**2) * evals

    evals_green = np.array([1/(1 + ev_rs_sq) for ev_rs_sq in evals_r_s_squared])
    evals_green_k = evals_green[0:k]
    Lambda_Green_k = np.diag(evals_green_k)

    modes = emodes[:, 0:k]
    modes_pinv = np.linalg.pinv(modes)
    vertexModelSC = modes @ Lambda_Green_k  @ modes_pinv

    np.fill_diagonal(vertexModelSC, 0)
    vertexModelSC[vertexModelSC < 0] = 0
    
    vertexModelSC = (vertexModelSC + vertexModelSC.T)/2

    if density != 1:
        vertexModelSC = utilities.threshold_symmetric_matrix_to_density(vertexModelSC, idxes_vertex, density=density)

    if resampling_weights  == "gaussian":
        vertexModelSC = utilities.resample_matrix(vertexModelSC)
    
    else:
        vertexModelSC /= np.max(vertexModelSC)

    return vertexModelSC

def generate_high_res_LBO_humans(r_s, k, emodes, evals, density, idxes_vertex, resampling_weights):

    evals_k = evals[0:k]
    Lambda = np.diag(evals_k)

    modes = emodes[:, 0:k]
    modes_pinv = np.linalg.pinv(modes)

    vertexModelSC = modes @ Lambda @ modes_pinv

    np.fill_diagonal(vertexModelSC, 0)
    vertexModelSC[vertexModelSC < 0] = 0
    
    vertexModelSC = (vertexModelSC + vertexModelSC.T)/2

    if density != 1:
        vertexModelSC = utilities.threshold_symmetric_matrix_to_density(vertexModelSC, idxes_vertex, density=density)

    if resampling_weights  == "gaussian":
        vertexModelSC = utilities.resample_matrix(vertexModelSC)
    
    else:
        vertexModelSC /= np.max(vertexModelSC)

    return vertexModelSC


def generate_parcellated_GEM_humans(r_s, k, emodes, evals, idxes_vertex, idxes_parcel, characteristic_matrix, fixed_threshold_vertex, n_edges_empirical, resampling_weights):
    
    evals_r_s_squared = (r_s**2) * evals

    evals_green = np.array([1/(1 + ev_rs_sq) for ev_rs_sq in evals_r_s_squared])
    
    evals_green_k = evals_green[0:k]
    Lambda_Green_k = np.diag(evals_green_k)

    modes = emodes[:, 0:k]
    modes_pinv = np.linalg.pinv(modes)
    vertexModelSC = modes @ Lambda_Green_k  @ modes_pinv

    np.fill_diagonal(vertexModelSC, 0)

    vertexModelSC[vertexModelSC < 0] = 0

    vertexModelSC = (vertexModelSC + vertexModelSC.T)/2

    vertexModelSC_thresholded = utilities.threshold_symmetric_matrix_to_density(vertexModelSC, idxes_vertex, density=fixed_threshold_vertex)

    model_parcellated  = utilities.downsample_high_resolution_structural_connectivity_to_atlas(vertexModelSC_thresholded,
                                                    characteristic_matrix)

    model_parcellated_thresholded = utilities.apply_threshold_to_match_densities(model_parcellated, n_edges_empirical, idxes_parcel)
    model_parcellated_thresholded /= np.max(model_parcellated_thresholded)

    if resampling_weights  == "gaussian":
        model_parcellated_thresholded = utilities.resample_matrix(model_parcellated_thresholded)

    return model_parcellated_thresholded

def generate_binary_network_from_p_distribution(p_ij, upper_tri_idx, n_vertices, n_connections_vertex, total_possible_connections):
    
    sampled_indices = np.argsort(np.random.rand(total_possible_connections) / p_ij)[:n_connections_vertex]

    row_idx = upper_tri_idx[0][sampled_indices]
    col_idx = upper_tri_idx[1][sampled_indices]

    adjacency_matrix = np.zeros((n_vertices, n_vertices), dtype=np.uint8)
        
    adjacency_matrix[row_idx, col_idx] = 1
    adjacency_matrix[col_idx, row_idx] = 1  

    return adjacency_matrix  

def generate_EDR_vertex_model(eta_prob_connection, eta_w, distances, idxes_vertex, n_vertices, n_edges_vertex_empirical, total_possible_connections, resampling_weights):

    p_ij = np.exp(-eta_prob_connection*distances)
    
    A_ij_binary_EDR = generate_binary_network_from_p_distribution(p_ij, idxes_vertex, n_vertices, n_edges_vertex_empirical, total_possible_connections)
    
    idxes_connections_model = np.nonzero(A_ij_binary_EDR[idxes_vertex])[0]
    distances_idxes_edges_model = distances[idxes_connections_model]
    
    vertexModelSC = np.zeros((n_vertices, n_vertices))
    vertexModelSC_idxes = np.zeros(len(idxes_vertex[0]))
    vertexModelSC_idxes[idxes_connections_model] = np.exp(-eta_w * distances_idxes_edges_model)
    
    vertexModelSC[idxes_vertex] = vertexModelSC_idxes
    vertexModelSC += vertexModelSC.T

    if resampling_weights  == "gaussian":
        vertexModelSC = resample_matrix(vertexModelSC)

    vertexModelSC /= np.max(vertexModelSC)

    return vertexModelSC

def generate_random_vertex_model(n_vertices, total_possible_connections, n_connections_vertex, idxes_vertex, weighted=False):
    idxe_random_edges = np.random.choice(total_possible_connections, size=n_connections_vertex, replace=False)
    vertexModelSC_idxes = np.zeros(total_possible_connections)
    if weighted == True:
        vertexModelSC_idxes[idxe_random_edges] = np.random.uniform(0.01, 1, size=len(idxe_random_edges))
    else:
        vertexModelSC_idxes[idxe_random_edges] = 1

    vertexModelSC_thresholded = np.zeros((n_vertices, n_vertices))
    vertexModelSC_thresholded[idxes_vertex] = vertexModelSC_idxes
    vertexModelSC_thresholded += vertexModelSC_thresholded.T

    return vertexModelSC_thresholded