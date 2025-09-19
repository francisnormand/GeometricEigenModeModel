import networkx as nx
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
from scipy.stats import pearsonr, spearmanr, ks_2samp
import seaborn as sns
# from graph_tool.all import *
from scipy.spatial.distance import pdist
from scipy.linalg import eigh
import re
from scipy.sparse import csr_matrix


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


def compute_node_properties(node_measures, connectivity_matrix, distances):
    
    binary_connectivity = np.copy(connectivity_matrix)
    binary_connectivity[binary_connectivity != 0 ] = 1

    dictionary_node_measures = {node_m:0 for node_m in node_measures}
    # graph_connectivity = to_graph_tool(binary_connectivity)
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
            # node_c = graph_tool.clustering.local_clustering(graph_connectivity)
            A = csr_matrix(binary_connectivity)
            degrees = np.array(A.sum(axis=1)).flatten()
            triangles = A @ A @ A
            triangles = triangles.diagonal() 
            clustering_coeffs = triangles / (degrees * (degrees - 1))
            dictionary_node_measures[node_measure] = clustering_coeffs

        elif node_measure == "ks_edge_distance":
            idxes_edges = np.nonzero(binary_connectivity[idxes_])[0]
            dictionary_node_measures[node_measure] = distances[idxes_edges]
            
        elif node_measure == "ranked_weights_strength":
            rank_weight_matrix = assignWeightBasedOnRank(connectivity_matrix, idxes_)
            dictionary_node_measures[node_measure] = np.sum(rank_weight_matrix, axis=1)    
    
    return dictionary_node_measures


def compute_node_average_connection_distance(distance_matrix, connectivity_matrix):
    

    binary_connection_times_distance = np.multiply(distance_matrix, connectivity_matrix)

    binary_connection_times_distance[np.nonzero(binary_connection_times_distance == 0)] = np.nan
    
    node_connection_distances = np.nanmean(binary_connection_times_distance, axis=1)
    

    nan_rows = np.isnan(node_connection_distances)
    node_connection_distances[nan_rows] = 0
    # print(node_connection_distances,"node_connection_distances")

    return node_connection_distances

