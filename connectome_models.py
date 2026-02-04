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

def generate_non_human_species_GEM(r_s, k, animal_parameters, resampling_weights=None):
    
    fixed_threshold_vertex = animal_parameters['fixed_threshold_vertex'] 
    n_edges_empirical = animal_parameters['n_edges_empirical_parcel']

    emodes = animal_parameters['emodes']
    evals = animal_parameters['evals']

    characteristic_matrix = animal_parameters['characteristic_matrix']
    vertices_in_connectome = animal_parameters['vertices_in_connectome']
    idxes_cortex = animal_parameters['idxes_cortex']

    n_parcels = characteristic_matrix.shape[0]

    idxes_parcels = np.triu_indices(n_parcels, k=1)

    if idxes_cortex is not None:
        emodes = emodes[idxes_cortex, :]

    if vertices_in_connectome is not None:
        n_vertices_connectome = len(np.nonzero(vertices_in_connectome)[0])
        print(n_vertices_connectome, "n_vertices_connectome")
        idxes_vertex_connectome = np.triu_indices(n_vertices_connectome, k=1)
    else:
        n_vertices = emodes.shape[0]
        idxes_vertex = np.triu_indices(n_vertices, k=1)
        idxes_vertex_connectome = idxes_vertex

    evals_r_s_squared = (r_s**2) * evals
    evals_green = 1/(1+evals_r_s_squared)

    evals_green_k = evals_green[0:k]
    Lambda_Green_k = np.diag(evals_green_k)

    modes = emodes[:, 0:k]
    modes_pinv = np.linalg.pinv(modes)

    if vertices_in_connectome is not None:
        modes = modes[vertices_in_connectome, :]
        modes_pinv = modes_pinv[:, vertices_in_connectome]

    vertexModelSC = modes @ Lambda_Green_k  @ modes_pinv

    np.fill_diagonal(vertexModelSC, 0)

    vertexModelSC[vertexModelSC < 0] = 0 

    vertexModelSC = (vertexModelSC + vertexModelSC.T)/2

    vertexModelSC_idxes = vertexModelSC[idxes_vertex_connectome]

    idxes_model = np.nonzero(vertexModelSC_idxes)[0]

    if vertices_in_connectome is not None:
        vertexModelSC_thresholded =  utilities.threshold_symmetric_matrix_to_density(vertexModelSC, idxes_vertex_connectome, density=fixed_threshold_vertex)
        
    else:
        vertexModelSC_thresholded = utilities.threshold_symmetric_matrix_to_density(vertexModelSC, idxes_vertex,  density=fixed_threshold_vertex)


    model_parcellated  = utilities.downsample_high_resolution_structural_connectivity_to_atlas(vertexModelSC_thresholded,
                                                    characteristic_matrix)


    model_parcellated_thresholded = utilities.apply_threshold_to_match_densities(model_parcellated, n_edges_empirical, idxes_parcels)
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
        vertexModelSC = utilities.resample_matrix(vertexModelSC)

    vertexModelSC /= np.max(vertexModelSC)

    return vertexModelSC


def generate_EDR_vertex_parcellated_model(eta_prob_connection, eta_w, distances_vertex, idxes_vertex, idxes_parcel, n_vertices, n_edges_vertex_empirical, total_possible_connections, resampling_weights, characteristic_matrix, n_edges_parcel_empirical):

    p_ij = np.exp(-eta_prob_connection * distances_vertex)
    
    A_ij_binary_EDR = generate_binary_network_from_p_distribution(p_ij, idxes_vertex, n_vertices, n_edges_vertex_empirical, total_possible_connections)
    
    idxes_connections_model = np.nonzero(A_ij_binary_EDR[idxes_vertex])[0]
    distances_idxes_edges_model = distances_vertex[idxes_connections_model]
    
    vertexModelSC = np.zeros((n_vertices, n_vertices))
    vertexModelSC_idxes = np.zeros(len(idxes_vertex[0]))
    vertexModelSC_idxes[idxes_connections_model] = np.exp(-eta_w * distances_idxes_edges_model)
    
    vertexModelSC[idxes_vertex] = vertexModelSC_idxes
    vertexModelSC += vertexModelSC.T

    if resampling_weights  == "gaussian":
        vertexModelSC = utilities.resample_matrix(vertexModelSC)

    modelSC = utilities.downsample_high_resolution_structural_connectivity_to_atlas(vertexModelSC, characteristic_matrix)
    modelSC = utilities.apply_threshold_to_match_densities(modelSC, n_edges_parcel_empirical, idxes_parcel)

    modelSC /= np.max(modelSC)

    return modelSC

def generate_distance_atlas_model(eta, nNodes, nConnectionsFORGRAPH, distanceMatrix_idxes, idxes, cost_rule):
   
    totalNumberOfEdges = len(idxes[0])
    #p_ij = np.exp(distanceMatrix[idxes]*eta)
    # p_ij = distanceMatrix[idxes]**eta
    
    p_ij = cost_rule(distanceMatrix_idxes, eta)

    p_ij /= np.sum(p_ij)
    E_uv = np.copy(p_ij)

    modelSC = np.zeros((nNodes,nNodes), dtype="float32")

    edgeIdxes = np.random.choice(totalNumberOfEdges, size=nConnectionsFORGRAPH, p=p_ij, replace=False)
    for edgeIdx in edgeIdxes:
        ii = idxes[0][edgeIdx]
        jj = idxes[1][edgeIdx]

        modelSC[ii, jj] = 1
        modelSC[jj, ii] = 1
    
    return modelSC

def update_matching(a, m=None, n=None, d=None, ii=None, jj=None):
    if m is None and n is None and d is None and ii is None and jj is None:
        assert np.array_equal(a.T, (a != 0).astype(float)), "a must be symmetric and of the correct type"
        n = 2 * np.dot(a, a)
        np.fill_diagonal(n, 0)
        temp = np.sum(a, axis=1)
        d = temp[:, None] + temp - 2 * a
        m = n / (d + (n == 0))
        return a, m, n, d

    # a[ii, jj] = 1
    # a[jj, ii] = 1

    temp1 = n[:, ii] + 2 * a[:, jj]
    temp2 = n[:, jj] + 2 * a[:, ii]
    
    n[:, ii] = temp1
    n[ii, :] = temp1
    n[:, jj] = temp2
    n[jj, :] = temp2
    n[ii, ii] = 0
    n[jj, jj] = 0

    temp = d[:, [ii, jj]] + 1
    d[:, [ii, jj]] = temp
    d[[ii, jj], :] = temp.T

    d[ii, jj] = d[ii, jj] - 1
    d[jj, ii] = d[jj, ii] - 1
    d[ii, ii] = d[ii, ii] + 1
    d[jj, jj] = d[jj, jj] + 1

    # print(n[:, [ii,jj]].shape, "n[:, [ii,jj]].shape")
    # print((n[:, [ii,jj]].T).shape, "n[:, [ii,jj]].shape")

    m[:, [ii, jj]] = n[:, [ii, jj]] / (d[:, [ii, jj]] + (n[:, [ii,jj]]== 0))
    m[[ii, jj], :] = n[[ii, jj], :] / (d[[ii, jj], :] + (n[[ii,jj], :] == 0))

    return a, m, n, d

def generate_matching_index_model(eta, gamma, nNodes, nConnectionsFORGRAPH, distanceMatrix_idxes, totalNumberOfEdges, idxes, cost_rule):
   
    p_ij = cost_rule(distanceMatrix_idxes, eta)

    p_ij /= np.max(p_ij)
    p_ij /= np.sum(p_ij)
    E_uv = np.copy(p_ij)

    modelSC = np.zeros((nNodes,nNodes), dtype="float32")
    edgesAdded = []
    
    for nConnectionToAdd in range(nConnectionsFORGRAPH):
        # print(nConnectionToAdd)
        edgeIdx = np.random.choice(totalNumberOfEdges, p=p_ij)
        
        ii = idxes[0][edgeIdx]
        jj = idxes[1][edgeIdx]

        modelSC[ii, jj] = 1
        modelSC[jj, ii] = 1
       
        if nConnectionToAdd == 0:
            _, matching_index_update, numerator, denominator = update_matching(modelSC)
        else:
            _, matching_index_update, numerator, denominator = update_matching(modelSC, matching_index_update, numerator, denominator, ii, jj)
        
        matching_index_diago = matching_index_update[idxes]
        matching_index_diago[edgeIdx] = 0

        matching_index_diago += 1e-6
        matching_index_diago /= np.max(matching_index_diago)
        
        matching_index_diago = matching_index_diago**gamma

        matching_index_diago /= np.max(matching_index_diago)
        
        p_ij = E_uv * matching_index_diago
        
        p_ij[edgeIdx] = 0
        p_ij /= np.sum(p_ij)

        E_uv[edgeIdx] = 0
        E_uv /= np.max(E_uv)

    return modelSC


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

def generate_random_parcellated_model(n_vertices, total_possible_connections, n_connections_vertex, idxes_vertex, characteristic_matrix, idxes_parcel, n_edges_empirical, resampling_weights, weighted=False):
    idxe_random_edges = np.random.choice(total_possible_connections, size=n_connections_vertex, replace=False)
    vertexModelSC_idxes = np.zeros(total_possible_connections)

    if weighted == True:
        vertexModelSC_idxes[idxe_random_edges] = np.random.uniform(0.01, 1, size=len(idxe_random_edges))
    else:
        vertexModelSC_idxes[idxe_random_edges] = 1

    vertexModelSC_thresholded = np.zeros((n_vertices, n_vertices))
    vertexModelSC_thresholded[idxes_vertex] = vertexModelSC_idxes
    vertexModelSC_thresholded += vertexModelSC_thresholded.T

    model_parcellated  = utilities.downsample_high_resolution_structural_connectivity_to_atlas(vertexModelSC_thresholded,
                                                    characteristic_matrix)

    model_parcellated_thresholded = utilities.apply_threshold_to_match_densities(model_parcellated, n_edges_empirical, idxes_parcel)
    model_parcellated_thresholded /= np.max(model_parcellated_thresholded)

    if resampling_weights  == "gaussian":
        model_parcellated_thresholded = utilities.resample_matrix(model_parcellated_thresholded)

    return model_parcellated_thresholded