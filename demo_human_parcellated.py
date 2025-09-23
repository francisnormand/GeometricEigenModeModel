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


def generate_geometric_modes(number_of_parcels=300):
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
        - human_parcelalted_{number_of_parcels}_evals_lump_{lump}_masked_{cortex_mask}.npy
        - human_parcelalted_{number_of_parcels}_emodes_lump_{lump}_masked_{cortex_mask}.npy
        - human__parcelalted_{number_of_parcels}_Bmatrix_lump_{lump}_masked_{cortex_mask}.npy (if masked)
    """
    lump = False

    cortex_mask = True
    # cortex_mask = False
  
    max_num_modes = 500

    surface, surface_path = utilities.get_human_template_surface()

    output_eval_filename = path_data + f"/Schaefer{number_of_parcels}/human_parcellated_evals_lump_{lump}_masked_{cortex_mask}.npy"
    output_emode_filename = path_data + f"/Schaefer{number_of_parcels}/human_parcellated_emodes_lump_{lump}_masked_{cortex_mask}.npy"
    output_B_matrix_filename = None # Not saving the B matrix

    if os.path.exists(output_emode_filename):
        print(f"{output_emode_filename}")
        print()
        print("geometric modes already exists and will be overwritten")
        print()

    if cortex_mask is False: 
        evals, emodes = utilities.calc_surface_eigenmodes_nomask(surface_path, output_eval_filename, output_emode_filename, max_num_modes)
    else:
        cortex_mask_array = utilities.get_human_parcellated_cortex_mask(path_data, number_of_parcels)
        save_cut = 0  #Not saving temporary cut surface       
        evals, emodes, B_matrix = utilities.calc_surface_eigenmodes(surface_path, cortex_mask_array, output_eval_filename, output_emode_filename, output_B_matrix_filename, save_cut=save_cut, num_modes=max_num_modes, lump=lump)
    
    print()
    print("geometric modes were saved")

# Current working director
cwd = os.getcwd()

# Global variable here. So it only is defined once. 
path_data = f"{cwd}/data/human_parcellated"

# Path to the Conda environment
conda_prefix = os.environ.get("CONDA_PREFIX")

env_path = conda_prefix.split("/conda/")[0]
# print(env_path, "base_conda_path")

# Current environment name
conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
# print(conda_env_name)


def mainFunction():
    """
    Main function to reproduce the analysis on the high-resolution human connectome.

    These functions have to be run sequentially.
    """

    # 1. Generate the geometric eigenmodes
    generate_geometric_modes()

    # 2. Optimize the GEM (explore parameters landscape)
    # optimize_and_save_human_parcellated_results()
    # print()

    #3. Visualize performance
    # visualize_GEM_human_parcellated_results()

    # results = "main"
    # results = "modularity"
    # results = "spectral"
    
    #4. Generate benchmark models
    # generate_human_parcellated_comparison_results(which_results=results)

    #5. Compare GEM performance with other models
    # compare_human_parcellated_models(which_results=results)


if __name__ == "__main__":
    mainFunction()