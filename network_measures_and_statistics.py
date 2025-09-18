import networkx as nx
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
# from scipy.sparse import csr_matrix, issparse, diags
import matplotlib.pyplot as plt
from numba import jit, njit
from scipy.stats import pearsonr, spearmanr, ks_2samp
import seaborn as sns
from graph_tool.all import *
from scipy.spatial.distance import pdist
from scipy.linalg import eigh
# from portrait_divergence import portrait_divergence, portrait_divergence_weighted
import re
# from numba import njit, jit

def compute_hamming_dist(sequence_1, sequence_2):
    return len(np.where(sequence_1 != sequence_2)[0])/len(sequence_1)

def getWeightedMeasures(modelSC, empiricalSCMatrix, idxes):
    common_indexes_weights = np.where((modelSC!=0) & (empiricalSCMatrix!=0))
    recon_acc = pearsonr(modelSC[idxes], empiricalSCMatrix[idxes])[0]
    recon_accuracy_spearmanr = spearmanr(modelSC[idxes], empiricalSCMatrix[idxes])[0]
    recon_accuracy_commonEdges = spearmanr(empiricalSCMatrix[common_indexes_weights],modelSC[common_indexes_weights])[0]
    recon_accuracy_LogNormCommonEdges = spearmanr(np.log(modelSC[idxes]), np.log(empiricalSCMatrix[idxes]))[0]

    return recon_acc, recon_accuracy_spearmanr, recon_accuracy_commonEdges, recon_accuracy_LogNormCommonEdges

def calculate_max_ks_and_scores(modelSC, distanceMatrix, propertiesListEmpirical, max_weight_empirical=False, binary=False):
    if binary == True:
        nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkPropertiesBinaryAlways(modelSC, distanceMatrix)
    else:
        nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkProperties(modelSC, distanceMatrix, max_weight_empirical, binary=False)
    
    [degreesEmpirical, values_betweennessEmpirical, values_clusteringEmpirical, edgeDistancesEmpirical] = propertiesListEmpirical

    ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
    ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
    ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]
    ksEdgeDistance = ks_2samp(edgeDistancesModel, edgeDistancesEmpirical)[0]

    max_ks = max([ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance])
    corr = spearmanr(degreesModel, degreesEmpirical)[0]
    
    return max_ks, corr, [ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance]

def calculate_score_voronoi(modelSC, distanceMatrix, propertiesListEmpirical, optimizedFor, empiricalSC_idxes, idxes, return_scores=False):

    [degreesEmpirical, values_betweennessEmpirical, values_clusteringEmpirical, edgeDistancesEmpirical] = propertiesListEmpirical
    
    if optimizedFor == "ks":

        nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkPropertiesBinaryAlways(modelSC, distanceMatrix)
        ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
        ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
        ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]
        ksEdgeDistance = ks_2samp(edgeDistancesModel, edgeDistancesEmpirical)[0]
        score = np.max([ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance])

        return score

    elif optimizedFor == "degree":
        # modelSC += modelSC.T
        degreesModel = np.sum(modelSC, axis=1)
        
        return 1 - spearmanr(degreesModel, degreesEmpirical)[0]

    elif optimizedFor == "hamming":
        return compute_hamming_dist(modelSC[idxes], empiricalSC_idxes)

    elif optimizedFor == "hamming_and_degree":

        modelSC += modelSC.T
        degreesModel = np.sum(modelSC, axis=1)
        spearmanr_degree = spearmanr(degreesModel, degreesEmpirical)[0]
        hamming_distance = compute_hamming_dist(modelSC[idxes], empiricalSC_idxes)
        
        score = (1-spearmanr_degree) + hamming_distance

        if return_scores == True:
            return score, spearmanr_degree, hamming_distance
        else:
            return score

    elif optimizedFor == "ks_hamming_degree":
        nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkPropertiesBinaryAlways(modelSC, distanceMatrix)
        ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
        ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
        ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]
        ksEdgeDistance = ks_2samp(edgeDistancesModel, edgeDistancesEmpirical)[0]
        ks = np.max([ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance])

        hamming_distance = compute_hamming_dist(modelSC[idxes], empiricalSC_idxes)
        spearmanr_degree = spearmanr(degreesModel, degreesEmpirical)[0]

        score = ks + hamming_distance + (1 - spearmanr_degree)

        return score

def calculate_all_scores(modelSC, distanceMatrix, propertiesListEmpirical, empiricalSC_idxes, idxes):

    [degreesEmpirical, values_betweennessEmpirical, values_clusteringEmpirical, edgeDistancesEmpirical] = propertiesListEmpirical
    


    nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkPropertiesBinaryAlways(modelSC, distanceMatrix)
    ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
    ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
    ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]
    ksEdgeDistance = ks_2samp(edgeDistancesModel, edgeDistancesEmpirical)[0]
    
    max_ks = np.max([ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance])
        
    degree_corr = spearmanr(degreesModel, degreesEmpirical)[0]

    hamming_dist =  compute_hamming_dist(modelSC[idxes], empiricalSC_idxes)

    return max_ks, degree_corr, hamming_dist


def calculate_ks_scores_no_distance(modelSC, propertiesListEmpirical, empiricalSC_idxes, idxes):

    [degreesEmpirical, values_betweennessEmpirical, values_clusteringEmpirical] = propertiesListEmpirical
    


    [degreesModel, values_betweennessModel, values_clusteringModel] = computeBinaryNetworkProperties(modelSC)
    ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
    ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
    ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]

    ks_measures = {"ks_degree":ksDegrees, "ks_betweenness":ksBetweenness, "ks_clustering":ksClustering}
    return ks_measures


def calculate_all_binary_scores_dict(all_scores_heatMaps, modelSC, distanceMatrix, propertiesListEmpirical, empiricalSC_idxes, idxes, empiricalSCGraph_binary):
    scores_dict = {}

    [degreesEmpirical, values_betweennessEmpirical, values_clusteringEmpirical, edgeDistancesEmpirical] = propertiesListEmpirical
    
    nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkPropertiesBinaryAlways(modelSC, distanceMatrix)
    ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
    ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
    
    ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]
    ksEdgeDistance = ks_2samp(edgeDistancesModel, edgeDistancesEmpirical)[0]
    
    max_ks = np.max([ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance])
        
    degree_corr = spearmanr(degreesModel, degreesEmpirical)[0]

    hamming_dist =  compute_hamming_dist(modelSC[idxes], empiricalSC_idxes)

    portrait_div_binary = portrait_divergence(nx.from_numpy_array(modelSC), empiricalSCGraph_binary)

    for score in all_scores_heatMaps:
        if score=="max_ks":
            scores_dict[score] = max_ks
        elif score == "degreeCorr":
            scores_dict[score] = degree_corr
        elif score == "ks_degree":
            scores_dict[score] = ksDegrees
        elif score == "ks_betweenness":
            scores_dict[score] =ksBetweenness
        elif score == "ks_clustering":
            scores_dict[score] = ksClustering
        elif score == "ks_edgeDistance":
            scores_dict[score] = ksEdgeDistance
        elif score == "hamming":
            scores_dict[score] = hamming_dist
        elif score == "p_div":
            scores_dict[score] = portrait_div_binary

    return scores_dict

def calculate_all_weighted_scores_dict(all_scores_heatMaps, modelSC, distanceMatrix, propertiesListEmpirical, empiricalSC_idxes, idxes, max_weight_empirical):
    scores_dict = {}
    
    [degreesEmpirical, values_betweennessEmpirical, values_clusteringEmpirical, edgeDistancesEmpirical] = propertiesListEmpirical
    
    nEdgesModel, [degreesModel, values_betweennessModel, values_clusteringModel, edgeDistancesModel] = computeNetworkProperties(modelSC, distanceMatrix, max_weight_empirical, binary=False)
    ksDegrees = ks_2samp(degreesModel, degreesEmpirical)[0]
    ksBetweenness = ks_2samp(values_betweennessModel, values_betweennessEmpirical)[0]
    
    ksClustering = ks_2samp(values_clusteringModel, values_clusteringEmpirical)[0]
    ksEdgeDistance = ks_2samp(edgeDistancesModel, edgeDistancesEmpirical)[0]
    
    max_ks = np.max([ksDegrees, ksBetweenness, ksClustering, ksEdgeDistance])
        
    degree_corr = spearmanr(degreesModel, degreesEmpirical)[0]

    recon_acc = pearsonr(modelSC[idxes], empiricalSC_idxes)[0]

    for score in all_scores_heatMaps:
        if score=="max_ks_w":
            scores_dict[score] = max_ks
        elif score == "degreeCorr_w":
            scores_dict[score] = degree_corr
        elif score == "ks_degree_w":
            scores_dict[score] = ksDegrees
        elif score == "ks_betweenness_w":
            scores_dict[score] =ksBetweenness
        elif score == "ks_clustering_w":
            scores_dict[score] = ksClustering
        elif score == "ks_edgeDistance_w":
            scores_dict[score] = ksEdgeDistance
        elif score == "recon":
            scores_dict[score] = recon_acc

    return scores_dict


def to_graph_tool(adj):
    num_nodes = adj.shape[0]  # Number of nodes
    g = graph_tool.Graph(directed=False)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights

    nnz = np.nonzero(np.triu(adj, 1))
    nedges = len(nnz[0])
    edges = np.hstack([np.transpose(nnz), np.reshape(adj[nnz], (nedges, 1))])

    # Add the nodes to the graph
    g.add_vertex(num_nodes)

    # Add the edges to the graph
    g.add_edge_list(edges, eprops=[edge_weights])

    return g


def to_numpy_array(graph):
    num_nodes = graph.num_vertices()
    adj = np.zeros((num_nodes, num_nodes), dtype=float)

    edge_weights = graph.edge_properties['weight']
    for edge in graph.edges():
        source = int(edge.source())
        target = int(edge.target())
        weight = edge_weights[edge]
        adj[source, target] = weight
        adj[target, source] = weight  # Assuming undirected graph

    return adj

def assignWeightBasedOnRank(weighted_matrix, idxes):
    n_vertices = weighted_matrix.shape[0]
    weighted_matrix_idxes = weighted_matrix[idxes]
    edges_idx  = np.nonzero(weighted_matrix_idxes)[0]
    edges_weights = weighted_matrix_idxes[edges_idx]

    sorted_weights_idx = np.argsort(edges_weights)
    ranks = np.empty_like(sorted_weights_idx)

    ranks[sorted_weights_idx] = np.arange(1, len(sorted_weights_idx) + 1)

    edges_weights_idx_as_rank = np.zeros(len(weighted_matrix_idxes))
    edges_weights_idx_as_rank[edges_idx] = ranks

    rank_weighted_matrix = np.zeros((n_vertices, n_vertices))
    rank_weighted_matrix[idxes] = edges_weights_idx_as_rank
    rank_weighted_matrix += rank_weighted_matrix.T

    return rank_weighted_matrix

# vertexModelSC_thresholded_idxes, vertexModelSC_thresholded_binary_idxes, empirical_vertex_connectivity_idxes, empirical_vertex_connectivity_binary_idxes, degreesEmpirical, degreesEmpiricalBinary, degreesModel, degreeModelBinary

def compute_recon(vertexSpaceSC_thresholded_idxes,  empirical_vertex_connectivity_idxes, *args):
    return fast_pearsonr(vertexSpaceSC_thresholded_idxes, empirical_vertex_connectivity_idxes)

def compute_degree(degreeModel, degreesEmpirical, *args):
    return spearmanr(degreeModel, degreesEmpirical)[0]

def compute_recon_spearmanr(vertexSpaceSC_thresholded_idxes, empirical_vertex_connectivity_idxes, *args):
    return fast_spearmanr_numba(vertexSpaceSC_thresholded_idxes, empirical_vertex_connectivity_idxes)

def compute_degreeBinary(degreesModelBinary, degreesEmpiricalBinary, *args):
    return spearmanr(degreesModelBinary, degreesEmpiricalBinary)[0]

def compute_hamming(vertexSpaceSC_thresholded_binary_idxes, empirical_vertex_connectivity_binary_idxes, *args):
    return len(np.where(vertexSpaceSC_thresholded_binary_idxes != empirical_vertex_connectivity_binary_idxes)[0])/len(empirical_vertex_connectivity_binary_idxes)

def compute_average_row_correlation(vertexSpaceSC_thresholded, low_rez_connectivity, *args):
    return np.mean([pearsonr(vertexSpaceSC_thresholded[row_i, :], low_rez_connectivity[row_i, :])[0] for row_i in range(low_rez_connectivity.shape[0])])

def compute_pearson_empirical_edges_correlation(vertexSpaceSC_thresholded_idxes, empirical_weighted_edges, idxes_edges_empirical , *args):
    return pearsonr(vertexSpaceSC_thresholded_idxes[idxes_edges_empirical], empirical_weighted_edges)[0]

def compute_spearman_empirical_edges_correlation(vertexSpaceSC_thresholded_idxes, empirical_weighted_edges, idxes_edges_empirical , *args):
    return spearmanr(vertexSpaceSC_thresholded_idxes[idxes_edges_empirical], empirical_weighted_edges)[0]

def compute_true_positive_rate(vertexSpaceSC_thresholded_binary_idxes, idxes_edges_empirical):
    return len(np.where(vertexSpaceSC_thresholded_binary_idxes[idxes_edges_empirical] != 0)[0])/len(idxes_edges_empirical)

def compute_pearson_union_weights(vertexSpaceSC_thresholded_idxes, empirical_vertex_connectivity_idxes, *args):
    idxes_union = np.where((vertexSpaceSC_thresholded_idxes != 0) | (empirical_vertex_connectivity_idxes != 0))[0]
    return fast_pearsonr(vertexSpaceSC_thresholded_idxes[idxes_union], empirical_vertex_connectivity_idxes[idxes_union])

def compute_spearman_union_weights(vertexSpaceSC_thresholded_idxes, empirical_vertex_connectivity_idxes, *args):
    idxes_union = np.where((vertexSpaceSC_thresholded_idxes != 0) | (empirical_vertex_connectivity_idxes != 0))[0]
    return fast_spearmanr_numba(vertexSpaceSC_thresholded_idxes[idxes_union], empirical_vertex_connectivity_idxes[idxes_union])

def compute_ranked_weights_strength(vertexSpaceSC_thresholded, empirical_ranked_weights_stength, *args):
    idxes_ = np.triu_indices(vertexSpaceSC_thresholded.shape[0])
    rank_weight_matrix_model = assignWeightBasedOnRank(vertexSpaceSC_thresholded, idxes_)
    
    return fast_spearmanr_numba(np.sum(rank_weight_matrix_model, axis=1),empirical_ranked_weights_stength)

def get_args_for_measure(network_measure_, *args):
    # Define which arguments each measure needs
    if network_measure_ == "recon" or network_measure_ == "recon_spearmanr" or network_measure_ == "pearson_union_weights" or network_measure_ == "spearman_union_weights":
        return args[0], args[2]
    elif network_measure_ == "degree":
        return args[6], args[4]
    elif network_measure_ == "degreeBinary":
        return args[7], args[5]
    elif network_measure_ == "hamming":
        return args[1], args[3]
    elif network_measure_ == "pearson_empirical_edges_correlation" or network_measure_ == "spearman_empirical_edges_correlation":
        return args[0], args[8], args[9]
    elif network_measure_ == "true_positive_rate":
        return args[1], args[9]
    elif network_measure_ == "ranked_weights_strength":
        return args[10], args[11]
    else:
        raise ValueError(f"Unknown network measure: {network_measure_}")


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
    # Mean center the data
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    # Compute the Pearson correlation manually
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    
    return numerator / denominator


def compute_node_properties(node_measures, connectivity_matrix, distances):
    
    binary_connectivity = np.copy(connectivity_matrix)
    binary_connectivity[binary_connectivity != 0 ] = 1

    dictionary_node_measures = {node_m:0 for node_m in node_measures}
    graph_connectivity = to_graph_tool(binary_connectivity)
    n_vertices = connectivity_matrix.shape[0]
    idxes_ = np.triu_indices(n_vertices, k=1)

    for node_measure in node_measures:
        if node_measure == "node connection distance":
            distance_matrix = np.zeros((n_vertices, n_vertices))
            distance_matrix[idxes_] = distances
            distance_matrix += distance_matrix.T
            
            dictionary_node_measures[node_measure] = compute_node_average_connection_distance(distance_matrix, binary_connectivity)
    
        elif node_measure == "degree":
            dictionary_node_measures[node_measure] = np.sum(connectivity_matrix, axis=1)
        
        elif node_measure == "degreeBinary":
            dictionary_node_measures[node_measure] = np.sum(binary_connectivity, axis=1)
            
        elif node_measure == "clustering":
            node_c = graph_tool.clustering.local_clustering(graph_connectivity)
            dictionary_node_measures[node_measure] = np.array(node_c.get_array())
        
        elif node_measure == "betweenness":
            node_b, edge_b = graph_tool.centrality.betweenness(graph_connectivity)
            dictionary_node_measures[node_measure] = np.array(node_b.get_array())

        elif node_measure == "ks_edge_distance":
            idxes_edges = np.nonzero(binary_connectivity[idxes_])[0]
            dictionary_node_measures[node_measure] = distances[idxes_edges]
            
        elif node_measure == "ranked_weights_strength":
            rank_weight_matrix = assignWeightBasedOnRank(connectivity_matrix, idxes_)
            dictionary_node_measures[node_measure] = np.sum(rank_weight_matrix, axis=1)    
    
    return dictionary_node_measures


def computeSimilarityMeasuresFromDict_new(network_measure_, *args):

    measure_functions = {
    "recon": compute_recon,
    "degree": compute_degree,
    "recon_spearmanr": compute_recon_spearmanr,
    "degreeBinary": compute_degreeBinary,
    "hamming": compute_hamming,
    "pearson_empirical_edges_correlation": compute_pearson_empirical_edges_correlation,
    "spearman_empirical_edges_correlation":compute_spearman_empirical_edges_correlation,
    "true_positive_rate" : compute_true_positive_rate,
    "pearson_union_weights" : compute_pearson_union_weights,
    "spearman_union_weights" : compute_spearman_union_weights,
    "ranked_weights_strength": compute_ranked_weights_strength,
    "spearman_clustering": compute_spearman_clustering,
    "spearman_node_connection_distance": compute_spearman_node_connection_distance
}

    if network_measure_ in measure_functions:
        # Get the right arguments for the specific measure
        specific_args = get_args_for_measure(network_measure_, *args)
        # Call the corresponding function with the specific arguments
        return measure_functions[network_measure_](*specific_args)
    else:
        raise ValueError(f"Unknown network measure: {network_measure_}")


def compute_node_average_connection_distance(distance_matrix, connectivity_matrix):
    

    binary_connection_times_distance = np.multiply(distance_matrix, connectivity_matrix)

    binary_connection_times_distance[np.nonzero(binary_connection_times_distance == 0)] = np.nan
    
    node_connection_distances = np.nanmean(binary_connection_times_distance, axis=1)
    

    nan_rows = np.isnan(node_connection_distances)
    node_connection_distances[nan_rows] = 0
    # print(node_connection_distances,"node_connection_distances")

    return node_connection_distances


def computeNetworkProperties(scMatrix, distanceMatrix, max_weight_empirical, binary=False):
    edges = np.nonzero(scMatrix)
    nEdges = len(edges[0])
    edgeDistances = [distanceMatrix[edges[0][i], edges[1][i]] for i in range(len(edges[0]))]
    scMatrix += scMatrix.T

    degrees = np.sum(scMatrix, axis=1)

    if binary==False:
        degrees /= np.sum(degrees)
    
    g = to_graph_tool(scMatrix)

    scMatrix_copy_clustering = np.copy(scMatrix)

    scMatrix_copy_clustering/=np.max(scMatrix_copy_clustering)
    scMatrix_copy_clustering *= max_weight_empirical

    g_clustering = to_graph_tool(scMatrix_copy_clustering)

    node_c = graph_tool.clustering.local_clustering(g_clustering, weight=g_clustering.edge_properties['weight'])
    values_clustering = np.array(node_c.get_array())

    if binary == False:
        edge_weights = g.edge_properties['weight']
        inverted_weights = 1.0 / edge_weights.a
        edge_weights.a = inverted_weights

    node_b, edge_b = graph_tool.centrality.betweenness(g, weight=g.edge_properties['weight'])
    values_betweenness = np.array(node_b.get_array())

    return nEdges, [degrees, values_betweenness, values_clustering, edgeDistances]

def computeNetworkPropertiesBinaryAlways(scMatrix, distanceMatrix):

    edges = np.nonzero(scMatrix)
    nEdges = len(edges[0])

    edgeDistances = [distanceMatrix[edges[0][i], edges[1][i]] for i in range(len(edges[0]))]

    scMatrix += scMatrix.T

    degrees = np.sum(scMatrix, axis=1)

    g = to_graph_tool(scMatrix)

    node_c = graph_tool.clustering.local_clustering(g, weight=g.edge_properties['weight'])
    values_clustering = np.array(node_c.get_array())


    node_b, edge_b = graph_tool.centrality.betweenness(g, weight=g.edge_properties['weight'])
    values_betweenness = np.array(node_b.get_array())

    return nEdges, [degrees, values_betweenness, values_clustering, edgeDistances]

def compute_binary_network_properties(scMatrix):

    degrees = np.sum(scMatrix, axis=1)

    g = to_graph_tool(scMatrix)

    node_c = graph_tool.clustering.local_clustering(g)
    values_clustering = np.array(node_c.get_array())


    node_b, edge_b = graph_tool.centrality.betweenness(g)
    values_betweenness = np.array(node_b.get_array())

    return [degrees, values_betweenness, values_clustering]

def compute_binary_network_properties_in_dict(scMatrix):

    degrees = np.sum(scMatrix, axis=1)

    g = to_graph_tool(scMatrix)

    node_c = graph_tool.clustering.local_clustering(g)
    values_clustering = np.array(node_c.get_array())


    node_b, edge_b = graph_tool.centrality.betweenness(g)
    values_betweenness = np.array(node_b.get_array())


    return  {"degreeBinary":degrees, "betweenness":values_betweenness, "clustering": values_clustering}