from matplotlib.colors import ListedColormap
import colormaps as cmaps_extra
import numpy as np
import lapy
from scipy.spatial.distance import pdist
import nibabel as nib
import numpy as np
import os
import sys
import brainspace.mesh as mesh
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr, linregress
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
import bct
import networkx as nx
from scipy.sparse.linalg import eigsh
import seaborn as sns
from scipy.linalg import eig
import pandas as pd
from network_measures_and_statistics import compute_node_properties, compute_true_positive_rate, fast_spearmanr_numba, fast_pearsonr

cwd = os.getcwd()

def resample_matrix(template, noise='gaussian', seed=None, rand_params=[0.5, 0.1], 
                    ignore_repeats=True, reset_zeros=True, resymmetrise=True):

    """
    Generates a matrix of noise in the same pattern as a template.
    
    Parameters:
    - template (np.ndarray): Input template matrix.
    - noise (str or np.ndarray): Type of noise ('gaussian', 'normal', 'uniform', 'integers', or custom array).
    - seed (int, optional): Random seed.
    - rand_params (list): Parameters for the random distribution. For 'gaussian', [mean, std]; 
                          for 'uniform', [min, max].
    - ignore_repeats (bool): Whether to ignore repeated values in the template.
    - reset_zeros (bool): Whether to reset zero entries in the output to zero.
    - resymmetrise (bool): Whether to resymmetrise the noise matrix if the template is symmetric.
    
    Returns:
    - spatial_noise (np.ndarray): The resampled noise matrix.
    """
    
    alpha = rand_params[0]  # mean or min of rand distribution
    beta = rand_params[1]   # SD or max of rand distribution

    # Set random seed if specified
    if seed is not None:
        np.random.seed(seed)

    # Make noise
    if ignore_repeats:
        u, uc = np.unique(template, return_inverse=True)
        n_rand = len(u)
    else:
        n_rand = template.size

    if isinstance(noise, str):
        if noise in ['gaussian', 'normal']:
            sorted_noise = np.sort(np.random.randn(n_rand) * beta + alpha)
        elif noise == 'uniform':
            sorted_noise = np.sort(np.random.rand(n_rand) * (beta - alpha) + alpha)
        elif noise == 'integers':
            sorted_noise = np.sort(np.random.randint(alpha, beta + 1, size=n_rand))
        else:
            raise ValueError("Invalid noise type")
            
    elif isinstance(noise, np.ndarray):

        assert noise.size == n_rand, "Noise should contain the same number of elements as input map"
        sorted_noise = np.sort(noise.flatten())
    else:
        raise ValueError("Noise input not valid")

    # Organise noise
    if not ignore_repeats:
        idx = np.argsort(template.flatten())
        spatial_noise = np.zeros_like(template.flatten())
        spatial_noise[idx] = sorted_noise
        spatial_noise = spatial_noise.reshape(template.shape)
    else:
        spatial_noise = sorted_noise[uc].reshape(template.shape)

    # Resymmetrise if necessary
    if resymmetrise and np.allclose(template, template.T):
        spatial_noise = np.triu(spatial_noise) + np.triu(spatial_noise).T - np.diag(np.diag(spatial_noise))

    # Reset zeros
    if reset_zeros:
        spatial_noise[template == 0] = 0

    return spatial_noise

def powerlawRule(ditanceArray, eta):
    return ditanceArray**eta

def exponentialRule(ditanceArray, eta):
    return np.exp(eta*ditanceArray)
    

def downsample_high_resolution_structural_connectivity_to_atlas(high_resolution_connectome,
                                                                parcellation):
    """
    From Sina Mansour
    
    Downsample the high-resolution structural connectivity matrix to the resolution of a brain atlas.

    Args:

        high_resolution_connectome: The high-resolution structural connectome (v x v sparse CSR matrix)

        parcellation: A p x v sparse percellation matrix (can also accept a soft parcellation)

    Returns:

        connectome: The atlas-resolution structural connectome.
    """
    return parcellation.dot(high_resolution_connectome.dot(parcellation.T))


def apply_threshold_to_match_densities(vertexSpaceSC, nEdgesVertexSpace, idxes):

    vertexSpaceSC_thresholded = np.zeros((vertexSpaceSC.shape[0], vertexSpaceSC.shape[0]))
    vertexSpaceSC_idxes = vertexSpaceSC[idxes]
    sorted_idx = np.argsort(vertexSpaceSC_idxes)[::-1]

    vertexSpaceSC_idxes[sorted_idx[int(nEdgesVertexSpace)::]] = 0
    vertexSpaceSC_thresholded[idxes] = vertexSpaceSC_idxes
    vertexSpaceSC_thresholded += vertexSpaceSC_thresholded.T

    del sorted_idx

    return vertexSpaceSC_thresholded

def get_colormap(reverse=True):

    original_cmap = cmaps_extra.ice 
    new_cmap = original_cmap(np.linspace(0, 0.93, 256))
    if reverse == True:
        new_cmap = new_cmap[::-1]  

    custom_cmap = ListedColormap(new_cmap)

    return custom_cmap

def get_custom_colormap(top_cmap=0.5):

    original_cmap = cmaps_extra.ice 
    new_cmap = original_cmap(np.linspace(0, top_cmap, 256))
    reversed_cmap = new_cmap[::-1]  
    custom_cmap = ListedColormap(reversed_cmap)

    return custom_cmap

def get_human_template_surface(path_surface_high_res=None):

    extension = "L.midthickness.5k.surf.vtk"

    if path_surface_high_res == None:
        path_surface_high_res = f"{cwd}/data/human_high_res"
    
    fullpath = path_surface_high_res+f"/{extension}"
    return lapy.TriaMesh.read_vtk(fullpath), fullpath

def get_human_cortex_mask(path):

    extension = "cortex_mask_L.midthickness.5k.surf"
    cortex_mask = np.loadtxt(path+f"/{extension}")
    return cortex_mask.astype(int)


def get_human_parcellated_cortex_mask(path, number_of_parcels):
    
    cortex_mask = np.load(path + f"/Schaefer{number_of_parcels}/cortex_mask_to_SC{number_of_parcels}.npy")

    return cortex_mask.astype(int)

def get_non_human_species_mesh(path_data, species):

    if species == "Mouse":
        filename_mesh = "rh_tet_mesh.vtk"
        full_path = f"{path_data}/{species}/{filename_mesh}"
        mesh = lapy.TetMesh.read_vtk(full_path)
        
    elif species == "Marmoset":
        filename_mesh = "MBM_v3.0.1_midthickness-lh.vtk"
        full_path = f"{path_data}/{species}/{filename_mesh}"
        mesh = lapy.TriaMesh.read_vtk(full_path)
        
    elif species == "Macaque":
        filename_mesh = "MacaqueYerkes19_10k_midthickness-lh.vtk"
        full_path = f"{path_data}/{species}/{filename_mesh}"
        mesh = lapy.TriaMesh.read_vtk(full_path)

    return mesh, full_path

def get_parcellated_human_centroids(number_of_parcels, path_data):

    extension_centroids = "Schaefer2018_{}Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv".format(number_of_parcels)
    path_centroids = f"{path_data}/Schaefer{number_of_parcels}/{extension_centroids}"
    
    centroids  = np.loadtxt(path_centroids, delimiter=",", dtype=str)
    centroids = centroids[1:, 2:]
    centroids = np.asarray(centroids, dtype=int)
    L_hemi_idx = int(number_of_parcels/2)
    centroids = centroids[0:L_hemi_idx, :]
    print(centroids.shape, "centroids shape")
    distances = pdist(centroids)

    return distances, centroids

def get_vertex_human_centroids(surface_name, connectome_origin):

    path_base_save_thresholded_connectome = f"/fs03/kg98/FrancisN/connectomes/{connectome_origin}"
    path_base_surface = "/home/fnormand/kg98/Arshiya/data/hpc_highres/sina_connectome"

    surface = path_base_surface + "/" + f"{surface_name}" + ".gii"

    if "S1200" in surface_name:
        file_path_cortex_mask = "/home/fnormand/kg98_scratch/FrancisN/data/atlases/human_atlases/template_surfaces/fslr32k/fsLR_32k_cortex-lh_mask.txt"
    
    else:
        file_path_cortex_mask = path_base_save_thresholded_connectome + "/cortex_mask_{}.txt".format(surface_name.replace(".surf", ''))

    cortex_mask = np.loadtxt(file_path_cortex_mask)

    idxes_cortex = np.where(cortex_mask == 1)[0]
    
    vertices = nib.load(surface).darrays[0].data
    vertices_cortex = vertices[idxes_cortex, :]
    distances = pdist(vertices_cortex)

    return distances, idxes_cortex

def get_non_human_species_centroids(species, path_data):

    if species == "Marmoset":
        path_centroids = f"{path_data}/{species}/centroids_parcels_connectome.npy"
        centroids = np.load(path_centroids)	

    elif species == "Macaque":
        path_centroids = f"{path_data}/{species}/centroids_connectome.npy"
        centroids = np.load(path_centroids)	

    elif species == "Mouse":
        path_centroids = f"{path_data}/{species}/rh_centroids.txt"
        centroids = np.loadtxt(path_centroids, delimiter=",")	

    distances = pdist(centroids)
    return distances, centroids


def get_human_empirical_parcellated_connectome(path, target_density=0.1, number_of_parcels=300, connectome_type="smoothed"):

    if connectome_type == "smoothed":
        extension_connectome = f"from_32k_raw_template_smooth_fwhm=8_Schaefer{number_of_parcels}_connectome.npy"

    elif connectome_type == "unsmoothed":
        extension_connectome = f"from_32k_raw_template_connectome_Schaefer{number_of_parcels}.npy"
  
    empirical_parcel_connectivity = np.load(f"{path}/Schaefer{number_of_parcels}/{extension_connectome}")

    idxes_parcels = np.triu_indices(empirical_parcel_connectivity.shape[0], k=1)

    n_edges_threshold = int(target_density * len(idxes_parcels[0]))
    empirical_parcel_connectivity = threshold_symmetric_matrix_to_density(empirical_parcel_connectivity, idxes_parcels, target_density)
   
    empirical_parcel_connectivity /= np.max(empirical_parcel_connectivity)

    n_edges_empirical_parcel = len(np.nonzero(empirical_parcel_connectivity[idxes_parcels])[0])

    return empirical_parcel_connectivity, n_edges_empirical_parcel


def threshold_symmetric_matrix_to_density(matrix, triu_indices, density):

    triu_values = matrix[triu_indices]
    threshold_value = np.percentile(triu_values, 100 * (1 - density))
    
    thresholded_triu_values = np.where(triu_values >= threshold_value, triu_values, 0)
    
    thresholded_matrix = np.zeros_like(matrix)
    thresholded_matrix[triu_indices] = thresholded_triu_values
    
    thresholded_matrix += thresholded_matrix.T
    
    return thresholded_matrix


def get_human_empirical_vertex_connectome(path_connectome, connectome_type="smoothed", fwhm=8, target_density=0.046, resampling_weights=False, npz_=False):

    if "unsmoothed" in connectome_type:
        extension_connectome = "human_vertex_5k_raw_connectome.npy"
    else:
        extension_connectome  =f"human_vertex_5k_smoothed_connectome_fwhm={fwhm}.npy"

    if npz_ == False:
        empirical_vertex_connectivity = np.load(path_connectome + "/" + extension_connectome)
    else:
        empirical_vertex_connectivity = sparse.load_npz(path_connectome + "/" + extension_connectome).toarray()
 
    n_vertices = empirical_vertex_connectivity.shape[0]
    idxes_vertex = np.triu_indices(n_vertices, k=1)
    n_edges_threshold = int(target_density * len(idxes_vertex[0]))

    empirical_vertex_connectivity = threshold_symmetric_matrix_to_density(empirical_vertex_connectivity, idxes_vertex, density=target_density)

    if resampling_weights  == "gaussian":
        empirical_vertex_connectivity = resample_matrix(empirical_vertex_connectivity)

    else:
        empirical_vertex_connectivity /= np.max(empirical_vertex_connectivity)
    
    return empirical_vertex_connectivity


def plotHumanOptLandscape(heatMap_average_opt, best_params, xticks=False, yticks=False, xlabels="number of modes $k$", ylabels="$r_s$", top=False, title_specify="", top_show=False):
    fig, ax = plt.subplots(dpi=400)

    cmap = get_colormap()

    if top == True:
        if top_show == False:
            percentile_min = np.percentile(heatMap_average_opt, 60)
        else:
            percentile_min = np.percentile(heatMap_average_opt, top_show)
        
        masked_heatmap = np.ma.masked_where(heatMap_average_opt < percentile_min, heatMap_average_opt)

        background_color = "#FFFFFF"
        # ax.set_facecolor([0.95, 0.95, 0.95])
        ax.set_facecolor(background_color)
        im = ax.imshow(masked_heatmap, cmap=cmap, interpolation='nearest', vmin=percentile_min, vmax=heatMap_average_opt.max())#, aspect='auto')
        ax.set_aspect(2.1)
        
        cbar_top = plt.colorbar(im, ax=ax)
        thicks_cbar = np.linspace(percentile_min, np.max(heatMap_average_opt), 5)
        thicks_cbar = np.round(thicks_cbar, 3)
        cbar_top.set_ticks(thicks_cbar)
        cbar_top.ax.tick_params(labelsize=10)

    else:
        if reverse is True:
            ax = sns.heatmap(heatMap_average_opt, cmap=cmap, ax=ax)
        else:
            ax = sns.heatmap(heatMap_average_opt, cmap=cmap, ax=ax)
        
        cbar = ax.collections[0].colorbar
        thicks_cbar = np.linspace(np.min(heatMap_average_opt), np.max(heatMap_average_opt), 5)
        thicks_cbar = np.round(thicks_cbar, 3)
        cbar.set_ticks(thicks_cbar)
        cbar.ax.tick_params(labelsize=20)

    if xticks is not None:
            x_positions = np.linspace(0, len(xticks) - 1, min(5, len(xticks))).astype(int)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([np.round(xticks[i],2) for i in x_positions] if xlabels else [])

    if yticks is not None:
        y_positions = np.linspace(0, len(yticks) - 1, min(5, len(yticks))).astype(int)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([np.round(yticks[i],2) for i in y_positions] if ylabels else [])
    
    def format_coord(x, y):
        """Display x, y, and corresponding heatmap value."""
        row = int(y)
        col = int(x)
        if 0 <= row < heatMap_average_opt.shape[0] and 0 <= col < heatMap_average_opt.shape[1]:
            z = heatMap_average_opt[row, col]  
            return f"x={col}, y={row}, value={z:.3f}"
        else:
            return "x={:.2f}, y={:.2f}".format(x, y)

    ax.format_coord = format_coord

    if best_params is not False:
        rect = Rectangle((best_params[1], best_params[0]), 1, 1, 
                        linewidth=0.6, edgecolor='white', facecolor='none')
        ax.add_patch(rect)

def grab_human_vertex_heatmaps(optimization_metric_list, directory, network_measures, dimension_files, r_s_values_list_for_files, formulation, target_density, connectome_type, fwhm=None, plot_all=False, plot_opt=False):

    heatmaps_dict = {}
    missings = []
    k_range = np.arange(2, 200)
    for idx_, network_measure_ in enumerate(network_measures):
        # print(network_measure_, "network measure")
        heatMap_measure = np.zeros(dimension_files)
        for idx_r, r_s in enumerate(r_s_values_list_for_files):
            if "unsmoothed" in connectome_type:
                extension_filename = f"{network_measure_}_formulation={formulation}_target_density={target_density}_r_s_id={idx_r}.npy"
            else:
                extension_filename = f"{network_measure_}_formulation={formulation}_fwhm={fwhm}_target_density={target_density}_r_s_id={idx_r}.npy"

            full_path = directory + "/" + extension_filename
            
            if os.path.exists(full_path):
                heatMap_measure[idx_r, :] = np.load(full_path)
            else:
                print(full_path, "does not exists")
                print(idx_r, "idx r missing")
                missings.append(idx_r)
        
        heatmaps_dict[network_measure_] = heatMap_measure

    if len(missings) > 0:
        print(missings, "missings")    
    
    list_of_heatmaps = []
    for optimization_metric in optimization_metric_list:
        list_of_heatmaps.append(heatmaps_dict[optimization_metric])

    average_heatmap = np.mean(np.array(list_of_heatmaps), axis=0)
    args_optimal = np.where(average_heatmap == np.max(average_heatmap))
    # print(args_optimal, "args_optimal")

    if plot_all == True:
        plotAllLanscapes(average_heatmap, k_range, r_s_values_list_for_files, top=True)

    if plot_opt == True:
        plotHumanOptLandscape(average_heatmap, args_optimal, xticks=k_range, yticks=r_s_values_list_for_files, xlabels="number of modes $k$", ylabels="$r_s$", top=True, title_specify="")

    return heatmaps_dict, args_optimal


def grab_human_vertex_LBO_heatmaps(optimization_metric_list, directory, network_measures, formulation, target_density, connectome_type, fwhm=None):

    heatmaps_dict = {}
    missings = []
    for idx_, network_measure_ in enumerate(network_measures):
        # print(network_measure_, "network measure")
        if "unsmoothed" in connectome_type:
            extension_filename = f"{network_measure_}_formulation={formulation}_target_density={target_density}.npy"
        else:
            extension_filename = f"{network_measure_}_formulation={formulation}_fwhm={fwhm}_target_density={target_density}.npy"

        full_path = directory + "/" + extension_filename
        
        if os.path.exists(full_path):
            heatMap_measure = np.load(full_path)
        else:
            print(full_path, "does not exists")
            
        
        heatmaps_dict[network_measure_] = heatMap_measure

    if len(missings) > 0:
        print(missings, "missings")    
    
    list_of_heatmaps = []
    for optimization_metric in optimization_metric_list:
        list_of_heatmaps.append(heatmaps_dict[optimization_metric])

    average_heatmap = np.mean(np.array(list_of_heatmaps), axis=0)
    args_optimal = np.where(average_heatmap == np.max(average_heatmap))

    return heatmaps_dict, args_optimal



def grab_human_EDR_heatmaps(repet_id, optimization_metric_list, directory, network_measures, dimension_files, eta_prob_list, formulation, target_density, connectome_type, fwhm=None, plot_heatmaps=False):
    
    heatmaps_dict = {}
    missings = []
    for idx_, network_measure_ in enumerate(network_measures):
        # print(network_measure_, "network measure")
        heatMap_measure = np.zeros(dimension_files)
        for idx_eta_prob, eta_prob in enumerate(eta_prob_list):

            if connectome_type == "smoothed":
                extension_filename = f"{network_measure_}_formulation={formulation}_fwhm={fwhm}_target_density={target_density}_eta_prob_conn_id_{idx_eta_prob}_repetition_id_{repet_id}.npy"
            else:
                extension_filename = f"{network_measure_}_formulation={formulation}_target_density={target_density}_eta_prob_conn_id_{idx_eta_prob}_repetition_id_{repet_id}.npy"

            full_path = directory + "/" + extension_filename
            
            if os.path.exists(full_path):
                # print(idx_eta_prob, "idx_eta_prob")
                heatMap_measure[idx_eta_prob, :] = np.load(full_path)
                
            else:
                print(full_path, "does not exists")
                print(idx_eta_prob, "idx_eta_prob missing")
                sys.exit()
                missings.append(idx_eta_prob)
        
        heatmaps_dict[network_measure_] = heatMap_measure

    if len(missings) > 0:
        print(missings, "missings")    

    list_of_heatmaps = []
    if optimization_metric_list[0] == "max_ks":
        for optimization_metric in ["ks_degree", "ks_clustering", "ks_edge_distance", "ks_betweenness"]:
            list_of_heatmaps.append(heatmaps_dict[optimization_metric])

        average_heatmap = np.max(np.array(list_of_heatmaps), axis=0)
        args_optimal = np.where(average_heatmap == np.min(average_heatmap))

    else:
        for optimization_metric in optimization_metric_list:
            list_of_heatmaps.append(heatmaps_dict[optimization_metric])

        average_heatmap = np.mean(np.array(list_of_heatmaps), axis=0)
        args_optimal = np.where(average_heatmap == np.max(average_heatmap))

    return heatmaps_dict, average_heatmap


def grab_human_distance_atlas_or_MI_heatmaps(optimization_metric_list, directory, formulation, target_density, connectome_type, fwhm, repetition_id, plot_heatmaps=False):
    
    heatmaps_dict = {}
    missings = []
    for idx_, network_measure_ in enumerate(optimization_metric_list):

        current_hypothesis = f"formulation={formulation}_fwhm={fwhm}_target_density={target_density}_repetition_id_{repetition_id}"
        full_path = directory + f"/{network_measure_}_{current_hypothesis}.npy"
        
        if os.path.exists(full_path):
            heatMap_measure = np.load(full_path)
        else:
            print(full_path, "does not exists")
            sys.exit()
            missings.append(idx_eta_prob)
        heatmaps_dict[network_measure_] = heatMap_measure

    if len(missings) > 0:
        print(missings, "missings")    
    
    list_of_heatmaps = []
    for optimization_metric in optimization_metric_list:
        list_of_heatmaps.append(heatmaps_dict[optimization_metric])

    average_heatmap = np.mean(np.array(list_of_heatmaps), axis=0)

    return average_heatmap, heatmaps_dict


def get_performance_results(network_measures, vertexModelSC, vertexModelSC_thresholded_idxes, empirical_vertex_connectivity_idxes, empirical_node_properties_dict, distances):

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

def plotConnectivity(connectivity_matrix, idxes_, title_="none", binary=False, figsize=False, original_cmap=plt.cm.cool, show_cbar=False):

    background_color = "#FFFFFF"

    custom_cmap = original_cmap
    
    masked_data = np.ma.masked_where(connectivity_matrix == 0, connectivity_matrix)

    nnz_weights = np.nonzero(connectivity_matrix[idxes_])[0]

    min_weight = np.min(connectivity_matrix[idxes_][nnz_weights])
    max_weight = np.max(connectivity_matrix[idxes_][nnz_weights])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=400)
    ax.set_facecolor(background_color)
    im = ax.imshow(masked_data, cmap=custom_cmap, interpolation='nearest', norm=LogNorm(vmin=min_weight, vmax=max_weight))

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_cbar is True:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)  # adjust pad to move it closer/farther
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)

def plot_human_vertex_scatter_splots(scatter_measures, vertexModelSC, vertexModelSC_thresholded_idxes, empirical_vertex_connectivity_idxes, node_properties_model_dict, empirical_node_properties_dict, distances, measure_colors, scatter_node_size=20, scatter_edge_size=5, alpha_node=0.6, alpha_edge=0.02, resampling_weights=None):

    if alpha_edge == 0.02:
        linewidth_edge = 0.1
    else:
        linewidth_edge = 0.6

    if alpha_node == 0.6:
        linewidth_node = 0.6
    else:
        linewidth_node = 1

    for measure_ in scatter_measures:
        plt.figure(dpi=150)
        print(measure_, "measure")

        if measure_ == "common_weights":
            idxes_common = np.where((vertexModelSC_thresholded_idxes != 0) & (empirical_vertex_connectivity_idxes != 0))[0]
            if resampling_weights == "gaussian":
                x = empirical_vertex_connectivity_idxes[idxes_common]
                y = vertexModelSC_thresholded_idxes[idxes_common]
            else:
                x = np.log10(empirical_vertex_connectivity_idxes[idxes_common])
                y = np.log10(vertexModelSC_thresholded_idxes[idxes_common])

            corr_, pval_ = spearmanr(x, y)

            plt.scatter(
                x, y,
                color=measure_colors[measure_],
                alpha=alpha_edge,
                s=scatter_edge_size,
                edgecolors='white',
                linewidth=linewidth_edge
            )

        else:
            x = empirical_node_properties_dict[measure_]
            y = node_properties_model_dict[measure_]
            corr_, pval_ = spearmanr(x, y)

            plt.scatter(
                x, y,
                color=measure_colors[measure_],
                alpha=alpha_node,
                s=scatter_node_size,
                edgecolors='white',
                linewidth=linewidth_node
            )

        slope, intercept, _, _, _ = linregress(x, y)
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color='black', linestyle="--", linewidth=1, alpha=0.7, zorder=10, label='_nolegend_')

        # Legend showing only rho and p
        legend_label = r"$\rho$ = {:.2f}, $p$ = {:.1e}".format(corr_, pval_)
        empty_handle = mlines.Line2D([], [], color='none')
        plt.legend([empty_handle], [legend_label], frameon=False, handlelength=0, handletextpad=0, fontsize=18)

        plt.title(measure_)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

        plt.tight_layout()

##### Generate geometric eigenmodes

def get_indices(surface_original, surface_new):
    """Extract indices of vertices of the two surfaces that match.

    Parameters
    ----------
    surface_original : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_new : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh

    Returns
    ------
    indices : array
        indices of vertices
    """

    indices = np.zeros([np.shape(surface_new.Points)[0],1])
    for i in range(np.shape(surface_new.Points)[0]):
        indices[i] = np.where(np.all(np.equal(surface_new.Points[i,:],surface_original.Points), axis=1))[0][0]
    indices = indices.astype(int)
    
    return indices

def create_temp_surface(surface_input, surface_output_filename):
    """Write surface to a new vtk file.

    Parameters
    ----------
    surface_input : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_output_filename : str
        Filename of surface to be saved
    """

    f = open(surface_output_filename, 'w')
    f.write('# vtk DataFile Version 2.0\n')
    f.write(surface_output_filename + '\n')
    f.write('ASCII\n')
    f.write('DATASET POLYDATA\n')
    f.write('POINTS ' + str(np.shape(surface_input.Points)[0]) + ' float\n')
    for i in range(np.shape(surface_input.Points)[0]):
        f.write(' '.join(map(str, np.array(surface_input.Points[i, :]))))
        f.write('\n')
    f.write('\n')
    f.write('POLYGONS ' + str(np.shape(surface_input.polys2D)[0]) + ' ' + str(4* np.shape(surface_input.polys2D)[0]) + '\n')
    for i in range(np.shape(surface_input.polys2D)[0]):
        f.write(' '.join(map(str, np.append(3, np.array(surface_input.polys2D[i, :])))))
        f.write('\n')
    f.close()


def calc_eig(tria, num_modes, lump):
    """Calculate the eigenvalues and eigenmodes of a surface.

    Parameters
    ----------
    tria : lapy compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    num_modes : int
        Number of eigenmodes to be calculated

    Returns
    ------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """
    
    fem = lapy.Solver(tria, lump=lump)
    evals, emodes = fem.eigs(k=num_modes)
    B_matrix = fem.mass.toarray()
    
    return evals, emodes, B_matrix

def calc_surface_eigenmodes(surface_input_filename, mask_input_filename, output_eval_filename, output_emode_filename, output_B_matrix_filename, save_cut, num_modes, lump=False):
    """Main function to calculate the eigenmodes of a cortical surface with application of a mask (e.g., to remove the medial wall).

    Parameters
    ----------
    surface_input_filename : str
        Filename of input surface
    mask_input_filename : str
        Filename of mask to be applied on the surface (e.g., cortex without medial wall, values = 1 for mask and 0 elsewhere)
    output_eval_filename : str  
        Filename of text file where the output eigenvalues will be stored
    output_emode_filename : str  
        Filename of text file where the output eigenmodes will be stored
    save_cut : boolean 
        Boolean to decide if the new surface with mask applied will be saved to a new surface file
    num_modes : int
        Number of eigenmodes to be calculated          
    """

    # load surface (as a brainspace object)
    surface_orig = mesh.mesh_io.read_surface(surface_input_filename)
    
    # load mask
    # can be any ROI (even whole cortex)
    if type(mask_input_filename) == np.ndarray:
        mask = mask_input_filename
    else:
        mask_input_file_main, mask_input_file_ext = os.path.splitext(mask_input_filename)
        if mask_input_file_ext == '.txt':
            mask = np.loadtxt(mask_input_filename)
        elif mask_input_file_ext == '.gii':
            mask = nib.load(mask_input_filename).darrays[0].data
    
    # create temporary suface based on mask
    
    surface_cut = mesh.mesh_operations.mask_points(surface_orig, mask)

    if save_cut == 1:
        # old method: save vtk of surface_cut and open via lapy TriaIO 
        # The writing phase of this process is very slow especially for large surfaces
        temp_cut_filename='temp_cut.vtk'
        create_temp_surface(surface_cut, temp_cut_filename)
        # load surface (as a lapy object)
        # tria = TriaIO.import_vtk(temp_cut_filename)
        tria = lapy.TriaMesh.read_vtk(surface_input_filename)
    else:
        # new method: replace v and t of surface_orig with v and t of surface_cut
        # faster version without the need to write the vtk file
        # load surface (as a lapy object)
        if "vtk" in surface_input_filename:
            # tria = TriaIO.import_vtk(surface_input_filename)
            tria = lapy.TriaMesh.read_vtk(surface_input_filename)
        elif "gii" in surface_input_filename:
            coords = nib.load(surface_input_filename).darrays[0].data
            faces = nib.load(surface_input_filename).darrays[1].data
            tria = lapy.TriaMesh(coords, faces)
        else:
            tria = mesh.mesh_io.read_surface(surface_input_filename)
        
        tria.v = surface_cut.Points
        tria.t = np.reshape(surface_cut.Polygons, [surface_cut.n_cells, 4])[:,1:4]

    if num_modes == None:
        num_modes = len(tria.v) - 1

    # calculate eigenvalues and eigenmodes
    evals, emodes, B_matrix = calc_eig(tria, num_modes, lump)
    
    # get indices of vertices of surface_orig that match surface_cut
    indices = get_indices(surface_orig, surface_cut)
    
    # reshape emodes to match vertices of original surface
    emodes_reshaped = np.zeros([surface_orig.n_points,np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[indices, mode] = np.expand_dims(emodes[:, mode], axis=1)

    np.save(output_eval_filename, evals)
    np.save(output_emode_filename, emodes_reshaped)
    if output_B_matrix_filename != None:
        np.save(output_B_matrix_filename, B_matrix)

    if save_cut == 0:
        if os.path.exists('temp_cut.vtk'):
            os.remove('temp_cut.vtk')

    return evals, emodes_reshaped, B_matrix

def calc_surface_eigenmodes_nomask(surface_input_filename, output_eval_filename, output_emode_filename, num_modes):
    """Main function to calculate the eigenmodes of a cortical surface without application of a mask.

    Parameters
    ----------
    surface_input_filename : str
        Filename of input surface
    output_eval_filename : str
        Filename of text file where the output eigenvalues will be stored
    output_emode_filename : str
        Filename of text file where the output eigenmodes will be stored
    num_modes : int
        Number of eigenmodes to be calculated
    """

    # load surface (as a lapy object)
    tria = lapy.TriaMesh.read_vtk(surface_input_filename)

    # calculate eigenvalues and eigenmodes
    if num_modes == None:
        num_modes = len(tria.v) - 1

    evals, emodes = calc_eig(tria, num_modes)

    # save eigenmode results
    np.save(output_eval_filename, evals)
    np.save(output_emode_filename, emodes)

    return evals, emodes

def newman_spectral_communities(G, k=None):
    B = nx.modularity_matrix(G)

    # print("got the B matrix")
    
    if k is None:
        k = estimate_optimal_k(B)  # Auto-detect k


    eigvals, eigvecs = eigsh(B, k=k, which="LA")  # Get k largest eigenvectors
    # print("got the eivals")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(eigvecs)
    
    # Assign communities to nodes
    nodes = list(G.nodes())
    return {nodes[i]: labels[i] for i in range(len(nodes))}

def efficient_newman_spectral_communities(G, list_of_number_of_communities):
    B = nx.modularity_matrix(G)

    max_k = np.max(list_of_number_of_communities)
    nodes = list(G.nodes())

    partition = {}
    eigvals, eigvecs = eigsh(B, k=max_k, which="LA")  # Get k largest eigenvectors

    for k in list_of_number_of_communities:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(eigvecs[:, 0:k])  
        partition[k] = {nodes[i]: labels[i] for i in range(len(nodes))}
    
    return partition

def labelsDict(G, partition_dict):
    labels_dict = {}

    for key_com, partition in partition_dict.items():
        labels_dict[key_com] = [partition[n] for n in G.nodes()]
    return labels_dict

def getDictOfPartitionsSet(partition_dict):
    partitions_set_dict = {}
    for key_com, partition in partition_dict.items():
        partitions_set_dict[key_com] = convertDicttoListOfBrackets(partition)
    
    return partitions_set_dict

def getPartition(G, k_part=None, method="Newman"):

    if method == "Newman":
        partition = newman_spectral_communities(G, k=k_part)

    else:
        # partition_empirical, q_emp = bct.modularity_louvain_und(connectome_mouse, seed=42)
        # partition_model = convertListToDict(partition_model)
        print("none implemented")
        sys.exit()
        pass

    return partition

def getDictOfPartitions(list_of_number_of_communities, G, method="Newman"):
    partition_dict = {}
    for n_com_key in list_of_number_of_communities:
        partition_dict[n_com_key] =  getPartition(G, k_part=n_com_key)

    return partition_dict

def convertDicttoListOfBrackets(partition):
    communities = {}

    for node, community in partition.items():
        if community not in communities:
            communities[community] = set()  
        communities[community].add(node)  

    list_of_sets = list(communities.values())

    return list_of_sets

def getDictOfNVI(labels_dict_empirical, labels_dict_model):
    nvi_dict = {}
    for key_com, labels_empirical in labels_dict_empirical.items():
        labels_model_ = labels_dict_model[key_com]

        nvi_dict[key_com], _ = bct.partition_distance(labels_empirical, labels_model_)
    
    return nvi_dict


#### Some code taken from: Suarez from https://github.com/netneurolab/suarez_connectometaxonomy/blob/main/scripts/1_spectral_%26_topological_feats/eigenfunctions.py

def eigen_spectrum(M):
    
    return np.abs(eig(M, left=False, right=False))

def norm_laplacian(A, degree='out'):

    D = np.zeros_like(A)

    np.fill_diagonal(D, np.sum(A, axis=1))

    # D_inv = 1.0 / D
    # D_inv[np.isinf(D_inv)] = 0
    
    D_inv = np.divide(1.0, D, out=np.zeros_like(D, dtype=float), where=D!=0)

    I = np.identity(len(D), dtype='int')

    L = I - D_inv @ A 
    
    return L

def compute_eigenspectrum(A, spectral_measure="normalized_laplacian"):

    if spectral_measure == "normalized_laplacian":
        L = norm_laplacian(A)
        eigs = eigen_spectrum(L)
    else:
        print("not implemented")
        sys.exit()

    return eigs

def compute_spectral_distance(spectra_1, spectra_2, metric="cosine"):
    array_spectras = np.vstack((spectra_1, spectra_2))
    return pdist(array_spectras, metric=metric)

def get_model_color_based_on_optimization_metric(model, measure, model_optimization_metrics, color_optimized, color_not_optimized):
    optimized_metrics = model_optimization_metrics.get(model, [])
    if measure in optimized_metrics:
        return color_optimized
    else:
        return color_not_optimized

def plot_all_measures_together(df, title_prefix, measure_colors, exclude_models_by_measure=None, different_opt_metrics=False, model_optimization_metrics=None, color_optimized=None, color_not_optimized=None, alpha=1):
    measures = df["measure"].unique()

    n_measures = len(measures)
    ncols = n_measures
    import math
    import numpy as np
    nrows = math.ceil(n_measures / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 6 * nrows), sharey=False)
    axes = axes.flatten()

    for i, measure in enumerate(measures):
        ax = axes[i]
        df_sub = df[df["measure"] == measure]

        if exclude_models_by_measure and measure in exclude_models_by_measure:
            excluded_models = exclude_models_by_measure[measure]
            df_sub = df_sub[~df_sub["model"].isin(excluded_models)]

        color = measure_colors.get(measure, "gray")
        unique_models = df_sub["model"].unique()

        if different_opt_metrics == False:
            palette = {model: color for model in unique_models}

        else:
            palette = {
                model: get_model_color_based_on_optimization_metric(
                    model,
                    measure,
                    model_optimization_metrics,
                    color_optimized,
                    color_not_optimized
                )
                for model in unique_models
            }

        sns.violinplot(
            data=df_sub,
            x="model",
            y="value",
            palette=palette,
            saturation=1,
            alpha=alpha,
            ax=ax
        )


        ax.set_title(f"{measure}")
        ax.set_xlabel("")

        y_min = df_sub["value"].min()
        y_max = df_sub["value"].max()
        y_ticks = np.linspace(y_min, y_max, 4)
        ax.set_yticks(y_ticks)
        ax.set_ylabel("")
        ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks], fontsize=24)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)


def linePlotModularity(list_of_number_of_communities, list_of_methods, results_dict, colors=None, ylabel=None):
    legend_handles = []
    # plt.figure(figsize=(6, 5), dpi=200)
    fig, ax = plt.subplots(figsize=(5, 6), dpi=200)

    for i_m, method in enumerate(list_of_methods):
        
        results_method = results_dict[method]
        means = {key: np.mean(value) for key, value in results_method.items()}
        std_devs = {key: np.std(value) for key, value in results_method.items()}
        
        x_values = sorted(means.keys())
        y_values = [means[key] for key in x_values]
        error_bars = [std_devs[key] for key in x_values]

        if colors is not None:
            color = colors[i_m]
        else:
            color = "darkblue"
        # handle = ax.errorbar(list_of_number_of_communities, y_values, yerr=error_bars, color=color, label=method, capsize=2,  marker='o', linestyle='-', alpha=0.85, linewidth=1, markersize=5, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8)
        handle = ax.scatter(list_of_number_of_communities, y_values, color=color, label=method,  marker='o', linestyle='-', alpha=0.85, edgecolors="lightgray", linewidth=1)
        legend_handles.append(handle)
        
        ax.plot(list_of_number_of_communities, y_values, color=color, label=method,  marker='', linestyle='-', alpha=0.85)

        ax.fill_between(
        x_values, 
        np.array(y_values) - np.array(error_bars), 
        np.array(y_values) + np.array(error_bars),  
        color=color, alpha=0.2  
    )

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel(f"{ylabel}")

    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()

    legend_fig, legend_ax = plt.subplots(figsize=(5, 5), dpi=200)
    legend_ax.axis("off")  

    legend = legend_ax.legend(
        # handles=legend_handles, labels=[method.capitalize() for method in list_of_methods], fontsize=12, loc="center"
        handles=legend_handles, labels=[method for method in list_of_methods], fontsize=12, loc="center"
    )

def boxPlotSpectral(title, df_net, colors):
    """
    Generate a box plot for spectral distance results grouped by model.

    Parameters:
    - title (str): Title of the plot
    - df_net (pd.DataFrame): DataFrame with columns 'model' and 'value',
                             where 'value' contains arrays of scores
    """
    plot_df = pd.DataFrame([
        {"model": row["model"], "value": val}
        for _, row in df_net.iterrows()
        for val in row["value"]
    ])
    
    models = df_net["model"].tolist()

    if colors is not None:
        if len(colors) != len(models):
            raise ValueError("Length of colors must match number of models.")
        palette = dict(zip(models, colors))
    else:
        palette = "Set2"

    plt.figure(figsize=(7, 5), dpi=150)
    sns.boxplot(x="model", y="value", data=plot_df, palette=palette, width=0.3, saturation=1, showfliers=False)
    
    sns.stripplot(x="model", y="value", data=plot_df, color="black", size=3.5, jitter=True, alpha=0.4, edgecolor='white', linewidth=0.4)

    plt.title(title)
    plt.xlabel("Model")
    plt.xticks()
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()